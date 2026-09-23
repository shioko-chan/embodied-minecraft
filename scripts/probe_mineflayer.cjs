"use strict";

// Use the open-source bot library on which Mindcraft depends.
const mineflayer = require("../.runtime/probe-node/node_modules/mineflayer");
const { Vec3 } = require("../.runtime/probe-node/node_modules/vec3");

const port = Number(process.env.MCSOCIETY_LAN_PORT || 55916);
const names = ["ProbeA", "ProbeB"];
const bots = names.map((username) => mineflayer.createBot({
  host: "127.0.0.1", port, username, version: "1.21", auth: "offline",
  checkTimeoutInterval: 60000,
}));

const timeout = setTimeout(() => {
  console.error("Timed out waiting for both Mineflayer bots to spawn");
  process.exitCode = 1;
  bots.forEach((bot) => bot.quit());
}, 30000);

Promise.all(bots.map((bot) => new Promise((resolve, reject) => {
  bot.once("spawn", () => resolve({
    username: bot.username,
    position: bot.entity.position.toArray(),
    version: bot.version,
  }));
  bot.once("kicked", (reason) => reject(new Error(`${bot.username} kicked: ${JSON.stringify(reason)}`)));
  bot.once("error", reject);
}))).then(async (spawned) => {
  clearTimeout(timeout);
  console.log(JSON.stringify({ mineflayer_spawned: spawned }));
  const explorer = bots[0];
  const start = explorer.entity.position.clone();
  explorer.setControlState("forward", true);
  await new Promise((resolve) => setTimeout(resolve, 1500));
  explorer.setControlState("forward", false);
  const end = explorer.entity.position.clone();
  const floor = explorer.blockAt(end.floored().offset(0, -1, 0));
  const distance = end.distanceTo(start);
  console.log(JSON.stringify({
    mineflayer_action: {
      start: start.toArray(), end: end.toArray(), distance,
      floor_block: floor?.name || null,
    },
  }));
  if (distance < 0.1 || !floor) throw new Error("Mineflayer movement/block query failed");
  let stone;
  for (let attempt = 0; attempt < 25; attempt++) {
    stone = explorer.inventory.items().find((item) => item.name === "stone");
    if (stone) break;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  if (!stone) throw new Error("Mineflayer did not observe the given stone item");
  await explorer.equip(stone, "hand");
  const support = explorer.blockAt(end.floored().offset(1, -1, 0));
  if (!support) throw new Error("No floor block adjacent to Mineflayer bot");
  await explorer.placeBlock(support, new Vec3(0, 1, 0));
  const placedPosition = support.position.offset(0, 1, 0);
  let placed;
  for (let attempt = 0; attempt < 25; attempt++) {
    placed = explorer.blockAt(placedPosition);
    if (placed?.name === "stone") break;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  console.log(JSON.stringify({
    mineflayer_placement: {
      position: placedPosition.toArray(), item_received: stone.name,
      block_observed: placed?.name || null,
    },
  }));
  if (placed?.name !== "stone") throw new Error("Mineflayer placement was not observed");
  let log;
  for (let attempt = 0; attempt < 25; attempt++) {
    log = explorer.inventory.items().find((item) => item.name === "oak_log");
    if (log) break;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  if (!log) throw new Error("Mineflayer did not observe the given oak log");
  const plankId = explorer.registry.itemsByName.oak_planks.id;
  const recipe = explorer.recipesFor(plankId, null, 1, null)[0];
  if (!recipe) throw new Error("No inventory oak-plank recipe found");
  await explorer.craft(recipe, 1, null);
  let plankCount = 0;
  for (let attempt = 0; attempt < 25; attempt++) {
    plankCount = explorer.inventory.items()
      .filter((item) => item.name === "oak_planks")
      .reduce((count, item) => count + item.count, 0);
    if (plankCount >= 4) break;
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  console.log(JSON.stringify({ mineflayer_crafting: {
    input: log.name, plank_count: plankCount,
    game_mode: explorer.game.gameMode,
    inventory: explorer.inventory.items().map((item) => `${item.name}:${item.count}`),
  } }));
  if (plankCount < 4) throw new Error("Mineflayer crafting did not yield four planks");
  await new Promise((resolve) => setTimeout(resolve, 1500));
  bots.forEach((bot) => bot.quit());
}).catch((error) => {
  clearTimeout(timeout);
  console.error(error);
  process.exitCode = 1;
  bots.forEach((bot) => bot.quit());
});
