// Exercise Mindcraft's existing bot initialization and high-level skills.
// The Python harness owns the real CraftGround world and grants test resources.
import { setSettings } from '../.runtime/upstream/mindcraft/src/agent/settings.js';
import { initBot } from '../.runtime/upstream/mindcraft/src/utils/mcdata.js';
import * as skills from '../.runtime/upstream/mindcraft/src/agent/library/skills.js';

setSettings({
  minecraft_version: '1.21',
  host: '127.0.0.1',
  port: Number(process.env.MCSOCIETY_LAN_PORT || 55916),
  auth: 'offline',
});

const bots = ['ProbeA', 'ProbeB'].map(initBot);
for (const bot of bots) {
  bot.output = '';
  // These skills only query cheat mode; the full Mindcraft mode scheduler is
  // deliberately outside this focused skill integration probe.
  bot.modes = { isOn: () => false };
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function waitForItem(bot, name) {
  for (let attempt = 0; attempt < 30; attempt++) {
    if (bot.inventory.items().some((item) => item.name === name)) return;
    await delay(200);
  }
  throw new Error(`${bot.username} did not receive ${name}`);
}

const timeout = setTimeout(() => {
  console.error('Timed out waiting for Mindcraft skills');
  process.exitCode = 1;
  bots.forEach((bot) => bot.quit());
}, 40000);

try {
  const spawned = await Promise.all(bots.map((bot) => new Promise((resolve, reject) => {
    bot.once('spawn', () => resolve({
      username: bot.username,
      position: bot.entity.position.toArray(),
      version: bot.version,
    }));
    bot.once('kicked', (reason) => reject(new Error(`${bot.username} kicked: ${reason}`)));
    bot.once('error', reject);
  })));
  console.log(JSON.stringify({ mineflayer_spawned: spawned }));

  const bot = bots[0];
  const start = bot.entity.position.clone();
  const destination = start.offset(5, 0, 0);
  const reached = await skills.goToPosition(
    bot, destination.x, destination.y, destination.z, 0.8,
  );
  const end = bot.entity.position.clone();
  console.log(JSON.stringify({
    mineflayer_action: {
      skill: 'Mindcraft.skills.goToPosition',
      reached, start: start.toArray(), end: end.toArray(),
      distance: end.distanceTo(start),
    },
  }));
  if (!reached || end.distanceTo(start) < 2) {
    throw new Error(`Mindcraft navigation failed: ${bot.output}`);
  }

  await waitForItem(bot, 'stone');
  await waitForItem(bot, 'oak_log');
  const crafted = await skills.craftRecipe(bot, 'oak_planks', 1);
  if (!crafted) throw new Error(`Mindcraft crafting failed: ${bot.output}`);
  console.log(JSON.stringify({
    mineflayer_crafting: {
      skill: 'Mindcraft.skills.craftRecipe',
      plank_count: bot.inventory.items()
        .filter((item) => item.name === 'oak_planks')
        .reduce((sum, item) => sum + item.count, 0),
    },
  }));

  const target = bot.entity.position.floored().offset(2, 0, 0);
  const placed = await skills.placeBlock(bot, 'stone', target.x, target.y, target.z);
  let observed = bot.blockAt(target)?.name;
  for (let attempt = 0; attempt < 20 && observed !== 'stone'; attempt++) {
    await delay(200);
    observed = bot.blockAt(target)?.name;
  }
  console.log(JSON.stringify({
    mineflayer_placement: {
      skill: 'Mindcraft.skills.placeBlock',
      position: target.toArray(), placed, block_observed: observed,
    },
  }));
  if (!placed || observed !== 'stone') {
    throw new Error(`Mindcraft block placement failed: ${bot.output}`);
  }
  await delay(1000);
} catch (error) {
  console.error(error);
  process.exitCode = 1;
} finally {
  clearTimeout(timeout);
  bots.forEach((bot) => bot.quit());
}
