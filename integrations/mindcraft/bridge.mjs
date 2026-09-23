// Thin JSON-line process adapter around pinned Mindcraft skills. This process
// joins an already published Minecraft world; it never starts or edits one.
import readline from 'node:readline';
import { setSettings } from '../../.runtime/upstream/mindcraft/src/agent/settings.js';
import { initBot } from '../../.runtime/upstream/mindcraft/src/utils/mcdata.js';
import * as skills from '../../.runtime/upstream/mindcraft/src/agent/library/skills.js';
import { Task } from '../../.runtime/upstream/mindcraft/src/agent/tasks/tasks.js';

const marker = 'MCSOCIETY:';
const send = (value) => process.stdout.write(`${marker}${JSON.stringify(value)}\n`);
const names = JSON.parse(process.env.MCSOCIETY_AGENT_NAMES || '["AgentA"]');
if (!Array.isArray(names) || names.length < 1 || names.length > 8 ||
    new Set(names).size !== names.length ||
    !names.every((name) => typeof name === 'string' && /^[A-Za-z0-9_]{1,16}$/.test(name))) {
  throw new Error('MCSOCIETY_AGENT_NAMES must be 1–8 unique Minecraft usernames');
}
setSettings({
  minecraft_version: '1.21',
  host: '127.0.0.1',
  port: Number(process.env.MCSOCIETY_LAN_PORT || 55916),
  auth: 'offline',
});

const bots = new Map(names.map((name) => [name, initBot(name)]));
const messages = new Map(names.map((name) => [name, []]));
let closing = false;
let activeTasks = null;
for (const [name, bot] of bots) {
  bot.output = '';
  // Skill functions query cheat mode. The full Mindcraft mode scheduler is not
  // started by this adapter; all exposed skills act in survival mode.
  bot.modes = {
    isOn: () => false,
    // No background mode scheduler runs in this skills-only adapter.
    pause: () => {},
    unpause: () => {},
  };
  bot.on('chat', (sender, content) => {
    const received = messages.get(name);
    received.push({ sender, content });
    if (received.length > 32) received.shift();
  });
  bot.on('kicked', (reason) => {
    if (!closing) send({ event: 'fatal', error: `Bot ${name} kicked: ${String(reason)}` });
  });
  bot.on('end', (reason) => {
    if (!closing) send({ event: 'fatal', error: `Bot ${name} disconnected: ${String(reason)}` });
  });
}

function state(name) {
  const bot = bots.get(name);
  const inventory = {};
  for (const item of bot.inventory.items()) {
    inventory[item.name] = (inventory[item.name] || 0) + item.count;
  }
  return {
    agent_id: name,
    position: bot.entity.position.toArray(),
    health: bot.health,
    food: bot.food,
    inventory,
    time: bot.time.timeOfDay,
    messages: [...messages.get(name)],
  };
}

async function execute(request) {
  const { id, agent, kind } = request;
  if (!Number.isSafeInteger(id) || !bots.has(agent)) {
    throw new Error('Invalid request id or agent');
  }
  const bot = bots.get(agent);
  if (kind === 'observe') {
    return { id, success: true, agents: Object.fromEntries(names.map((name) => [name, state(name)])) };
  }
  if (kind === 'block_at') {
    const [x, y, z] = request.position;
    const target = bot.entity.position.clone();
    target.x = x; target.y = y; target.z = z;
    return { id, success: true, block: bot.blockAt(target)?.name || null };
  }
  if (kind === 'configure_task') {
    if (activeTasks) throw new Error('MineCollab task is already configured');
    if (request.task?.type !== 'techtree' || request.task.agent_count !== names.length ||
        typeof request.task.task_id !== 'string') {
      throw new Error('Expected a supported MineCollab techtree task for the active agents');
    }
    activeTasks = names.map((name, count_id) => new Task(
      { name, count_id, bot: bots.get(name) }, structuredClone(request.task),
    ));
    return { id, success: true, task_id: request.task.task_id };
  }
  if (kind === 'evaluate_task') {
    if (!activeTasks) throw new Error('Configure a MineCollab task first');
    const agents = Object.fromEntries(activeTasks.map((task) => [
      task.name, task.validator.validate(),
    ]));
    return {
      id, success: Object.values(agents).some((score) => score.valid), agents,
      evidence_source: 'mindcraft_task_validator_mineflayer_inventory',
      server_verified: false,
    };
  }
  const before = state(agent);
  bot.output = '';
  let success;
  let observed_block = null;
  if (kind === 'navigate') {
    success = await skills.goToPosition(bot, ...request.position, request.radius);
  } else if (kind === 'craft') {
    success = await skills.craftRecipe(bot, request.item, request.count);
  } else if (kind === 'place_block') {
    const [x, y, z] = request.position;
    const target = bot.entity.position.clone();
    target.x = x; target.y = y; target.z = z;
    if (!bot.blockAt(target)) {
      throw new Error(`Target chunk is not loaded at ${target}`);
    }
    success = await skills.placeBlock(bot, request.block, ...request.position);
    observed_block = bot.blockAt(target)?.name || null;
  } else if (kind === 'deposit') {
    success = await skills.putInChest(bot, request.item, request.count);
  } else if (kind === 'withdraw') {
    success = await skills.takeFromChest(bot, request.item, request.count);
  } else if (kind === 'say') {
    bot.chat(request.message);
    success = true; // accepted for sending, not proof that a peer received it
  } else {
    throw new Error(`Unknown skill kind: ${kind}`);
  }
  return {
    id, agent, kind, success: success === true,
    before, after: state(agent),
    observed_block,
    log: bot.output,
  };
}

try {
  const timeout = setTimeout(() => {
    send({ event: 'fatal', error: 'Timed out waiting for Mindcraft bots to spawn' });
    process.exit(1);
  }, 30000);
  await Promise.all([...bots.values()].map((bot) => new Promise((resolve, reject) => {
    bot.once('spawn', resolve);
    bot.once('kicked', (reason) => reject(new Error(String(reason))));
    bot.once('end', (reason) => reject(new Error(String(reason))));
  })));
  clearTimeout(timeout);
  send({ event: 'ready', agents: names, minecraft: [...bots.values()][0].version });

  const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of input) {
    let id = null;
    try {
      const request = JSON.parse(line);
      id = request.id;
      send(await execute(request));
    } catch (error) {
      send({ id, error: { type: error.name, message: error.message, stack: error.stack } });
    }
  }
} catch (error) {
  send({ event: 'fatal', error: error.message });
  process.exitCode = 1;
} finally {
  closing = true;
  for (const bot of bots.values()) bot.quit();
}
