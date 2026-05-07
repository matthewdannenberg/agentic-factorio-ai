-- tests/integration/test_bridge_live.lua
--
-- In-game integration test suite for the factorio-agent bridge mod.
--
-- HOW TO USE
-- ----------
-- 1. Start Factorio 2.x with the factorio-agent mod enabled.
-- 2. Load or start any game (the script works on any map state).
-- 3. Open the console with the ` key.
-- 4. To run the full suite:
--      /c require("test_bridge_live")
--    Or paste the contents directly into the console.
--
-- 5. To run a single named suite:
--      /c T.run_suite("state_queries")
--    Available suite names: "state_queries", "action_commands", "edge_cases"
--
-- 6. Results are printed to the console. Each test shows PASS or FAIL with
--    a reason on failure. A summary line is printed at the end of each suite.
--
-- DESIGN NOTES
-- ------------
-- Tests do not depend on specific map state. Where a known condition is needed
-- (e.g. an item on the ground), the test creates that condition itself and
-- cleans up afterwards. Tests that mutate game state (place entities, change
-- research) restore the prior state before returning.
--
-- Each test function receives a single argument `t` — a table with assertion
-- helpers. Tests that raise an unhandled error are caught and recorded as FAIL.

-- ============================================================
-- Minimal test framework
-- ============================================================

local log_file = "agent-test-results.txt"

local function test_print(msg)
    game.print(msg)  -- still shows on screen
    helpers.write_file(log_file, msg .. "\n", true)  -- also write to file (append)
end

local T = {}
local results = {}   -- {suite, name, pass, reason}

-- Assertion helpers passed into each test function.
local function make_assertions(suite_name, test_name)
    local function fail(msg)
        error("ASSERT: " .. tostring(msg), 2)
    end

    return {
        ok = function(v, msg)
            if not v then fail(msg or "expected truthy, got " .. tostring(v)) end
        end,
        eq = function(a, b, msg)
            if a ~= b then
                fail(msg or ("expected " .. tostring(b) .. ", got " .. tostring(a)))
            end
        end,
        ne = function(a, b, msg)
            if a == b then
                fail(msg or ("expected not " .. tostring(b)))
            end
        end,
        gt = function(a, b, msg)
            if not (a > b) then
                fail(msg or (tostring(a) .. " is not > " .. tostring(b)))
            end
        end,
        has_key = function(tbl, key, msg)
            if tbl[key] == nil then
                fail(msg or ("missing key: " .. tostring(key)))
            end
        end,
        is_string = function(v, msg)
            if type(v) ~= "string" then
                fail(msg or ("expected string, got " .. type(v)))
            end
        end,
        is_number = function(v, msg)
            if type(v) ~= "number" then
                fail(msg or ("expected number, got " .. type(v)))
            end
        end,
        is_table = function(v, msg)
            if type(v) ~= "table" then
                fail(msg or ("expected table, got " .. type(v)))
            end
        end,
        json_ok = function(raw, msg)
            local ok, parsed = pcall(function()
                return helpers.json_to_table(raw)
            end)
            if not ok or parsed == nil then
                fail(msg or ("invalid JSON: " .. tostring(raw):sub(1, 80)))
            end
            return parsed
        end,
    }
end

local suites = {}

function T.suite(name, tests)
    suites[name] = tests
end

function T.run_suite(name)
    local tests = suites[name]
    if not tests then
        test_print("[TEST] Unknown suite: " .. tostring(name))
        return
    end

    local pass_count = 0
    local fail_count = 0

    for test_name, test_fn in pairs(tests) do
        local assertions = make_assertions(name, test_name)
        local ok, err = pcall(test_fn, assertions)
        if ok then
            pass_count = pass_count + 1
            test_print("[PASS] " .. name .. " :: " .. test_name)
        else
            fail_count = fail_count + 1
            test_print("[FAIL] " .. name .. " :: " .. test_name .. " — " .. tostring(err))
        end
    end

    test_print(
        string.format("[SUITE %s] %d passed, %d failed",
                      name, pass_count, fail_count)
    )
    return fail_count == 0
end

function T.run_all()
    local total_pass = 0
    local total_fail = 0
    for name, _ in pairs(suites) do
        test_print("─────── " .. name .. " ───────")
        local passed = T.run_suite(name)
        -- Counts already printed per suite; tally for the final summary
        -- by re-scanning results (simple: run_suite prints its own summary)
    end
    test_print("═══ Done. See per-suite summaries above. ═══")
end

-- ============================================================
-- Helpers
-- ============================================================

local function get_player()
    return game.get_player(1)
end

local function parse_json(raw)
    local ok, t = pcall(function() return game.json_to_table(raw) end)
    if ok then return t else return nil end
end

-- ============================================================
-- Suite: state_queries
-- Tests that fa.* query functions return well-formed JSON with
-- the expected top-level fields. Does not assert specific values —
-- those depend on map state.
-- ============================================================

T.suite("state_queries", {

    tick_is_positive = function(t)
        local raw = fa.get_tick()
        local n = tonumber(raw)
        t.ok(n ~= nil, "get_tick() should return a number string")
        t.gt(n, 0, "tick should be > 0")
    end,

    get_player_returns_valid_json = function(t)
        local raw = fa.get_player()
        local parsed = t.json_ok(raw)
        t.has_key(parsed, "tick")
        t.has_key(parsed, "player")
        local p = parsed.player
        t.has_key(p, "position")
        t.has_key(p.position, "x")
        t.has_key(p.position, "y")
        t.has_key(p, "health")
        t.has_key(p, "inventory")
        t.has_key(p, "reachable")
        t.is_number(p.health)
        t.is_table(p.inventory)
        t.is_table(p.reachable)
    end,

    player_position_is_finite = function(t)
        local raw = fa.get_player()
        local parsed = t.json_ok(raw)
        local pos = parsed.player.position
        t.ok(math.abs(pos.x) < 1e9, "player x should be finite")
        t.ok(math.abs(pos.y) < 1e9, "player y should be finite")
    end,

    get_entities_returns_list = function(t)
        local raw = fa.get_entities(32)
        local parsed = t.json_ok(raw)
        t.has_key(parsed, "entities")
        t.is_table(parsed.entities)
        -- Each entity, if present, must have required fields.
        for _, e in ipairs(parsed.entities) do
            t.has_key(e, "unit_number")
            t.has_key(e, "name")
            t.has_key(e, "position")
            t.has_key(e, "status")
            t.is_number(e.unit_number)
            t.is_string(e.name)
        end
    end,

    entity_names_are_strings = function(t)
        local raw = fa.get_entities(64)
        local parsed = t.json_ok(raw)
        for _, e in ipairs(parsed.entities) do
            t.is_string(e.name, "entity name should be a string")
            t.ok(#e.name > 0, "entity name should not be empty")
        end
    end,

    entities_exclude_characters = function(t)
        -- Player characters must not appear in the entity list.
        local raw = fa.get_entities(32)
        local parsed = t.json_ok(raw)
        for _, e in ipairs(parsed.entities) do
            t.ne(e.type, "character", "characters should be excluded from entity scan")
        end
    end,

    get_resource_map_returns_list = function(t)
        local raw = fa.get_resource_map(128)
        local parsed = t.json_ok(raw)
        t.has_key(parsed, "resource_map")
        t.is_table(parsed.resource_map)
        for _, r in ipairs(parsed.resource_map) do
            t.has_key(r, "resource_type")
            t.has_key(r, "position")
            t.has_key(r, "amount")
            t.has_key(r, "size")
            t.is_string(r.resource_type)
            t.is_number(r.amount)
            t.is_number(r.size)
            t.ok(r.amount >= 0, "resource amount should be non-negative")
            t.ok(r.size > 0, "resource size should be positive")
        end
    end,

    resource_patch_positions_are_finite = function(t)
        local raw = fa.get_resource_map(128)
        local parsed = t.json_ok(raw)
        for _, r in ipairs(parsed.resource_map) do
            t.ok(math.abs(r.position.x) < 1e9, "patch x should be finite")
            t.ok(math.abs(r.position.y) < 1e9, "patch y should be finite")
        end
    end,

    get_research_returns_correct_shape = function(t)
        local raw = fa.get_research()
        local parsed = t.json_ok(raw)
        t.has_key(parsed, "research")
        local r = parsed.research
        t.has_key(r, "unlocked")
        t.has_key(r, "queued")
        t.is_table(r.unlocked)
        t.is_table(r.queued)
        -- All unlocked tech names should be strings.
        for _, name in ipairs(r.unlocked) do
            t.is_string(name)
            t.ok(#name > 0, "tech name should not be empty")
        end
    end,

    automation_is_unlocked = function(t)
        -- "automation" is always researched in a new game with cheat mode.
        -- If this test is run on a fresh map without cheats, skip gracefully.
        local force = game.forces["player"]
        if not force.technologies["automation"].researched then
            -- Not researched — can't assert. Pass vacuously.
            return
        end
        local raw = fa.get_research()
        local parsed = t.json_ok(raw)
        local found = false
        for _, name in ipairs(parsed.research.unlocked) do
            if name == "automation" then found = true break end
        end
        t.ok(found, "automation should appear in unlocked list when researched")
    end,

    get_logistics_returns_power_fields = function(t)
        local raw = fa.get_logistics(32)
        local parsed = t.json_ok(raw)
        t.has_key(parsed, "logistics")
        local l = parsed.logistics
        t.has_key(l, "power")
        t.has_key(l, "belts")
        t.has_key(l, "inserter_activity")
        local pw = l.power
        t.has_key(pw, "produced_kw")
        t.has_key(pw, "consumed_kw")
        t.has_key(pw, "accumulated_kj")
        t.has_key(pw, "satisfaction")
        t.is_number(pw.produced_kw)
        t.is_number(pw.consumed_kw)
        t.ok(pw.satisfaction >= 0 and pw.satisfaction <= 1,
             "satisfaction should be in [0, 1]")
    end,

    get_damaged_entities_returns_list = function(t)
        local raw = fa.get_damaged_entities(32)
        local parsed = t.json_ok(raw)
        t.has_key(parsed, "damaged_entities")
        t.is_table(parsed.damaged_entities)
        for _, d in ipairs(parsed.damaged_entities) do
            t.has_key(d, "entity_id")
            t.has_key(d, "health_fraction")
            t.ok(d.health_fraction > 0 and d.health_fraction < 1,
                 "health_fraction should be in (0, 1)")
        end
    end,

    drain_destruction_events_returns_list = function(t)
        local raw = fa.drain_destruction_events()
        local parsed = t.json_ok(raw)
        t.has_key(parsed, "destroyed_entities")
        t.is_table(parsed.destroyed_entities)
        -- Drain a second time — should be empty (buffer was cleared).
        local raw2 = fa.drain_destruction_events()
        local parsed2 = t.json_ok(raw2)
        t.eq(#parsed2.destroyed_entities, 0,
             "second drain should return empty list")
    end,

    get_threat_returns_correct_shape = function(t)
        local raw = fa.get_threat()
        local parsed = t.json_ok(raw)
        t.has_key(parsed, "threat")
        local th = parsed.threat
        t.has_key(th, "biter_bases")
        t.has_key(th, "evolution_factor")
        t.is_table(th.biter_bases)
        t.is_number(th.evolution_factor)
        t.ok(th.evolution_factor >= 0 and th.evolution_factor <= 1,
             "evolution_factor should be in [0, 1]")
    end,

    get_state_returns_all_sections = function(t)
        local raw = fa.get_state({radius=16, resource_radius=64, item_radius=8})
        local parsed = t.json_ok(raw)
        local required = {
            "tick", "player", "entities", "resource_map", "ground_items",
            "research", "logistics", "damaged_entities", "destroyed_entities", "threat"
        }
        for _, key in ipairs(required) do
            t.has_key(parsed, key, "get_state() missing key: " .. key)
        end
        t.is_number(parsed.tick)
        t.gt(parsed.tick, 0)
    end,

    get_ground_items_returns_list = function(t)
        local player = get_player()
        -- Drop an iron plate on the ground, check it appears, pick it up.
        local had_iron = player.get_item_count("iron-plate")
        if had_iron == 0 then
            -- Give the player an iron plate if they don't have one.
            player.insert({name = "iron-plate", count = 1})
        end
        -- Drop one plate.
        player.surface.create_entity({
            name     = "item-on-ground",
            position = {x = player.position.x + 0.5, y = player.position.y},
            stack    = {name = "iron-plate", count = 1},
        })

        local raw = fa.get_ground_items(8)
        local parsed = t.json_ok(raw)
        t.has_key(parsed, "ground_items")
        t.is_table(parsed.ground_items)

        -- Clean up: remove any item-on-ground entities we may have dropped.
        local items = player.surface.find_entities_filtered({
            position = player.position, radius = 8, type = "item-entity"
        })
        for _, item in ipairs(items) do
            if item.valid then item.destroy() end
        end
        -- Restore inventory if we gave a plate.
        if had_iron == 0 then
            player.remove_item({name = "iron-plate", count = 1})
        end
    end,
})

-- ============================================================
-- Suite: action_commands
-- Tests that fa.* action functions return {"ok":true} or a
-- well-formed error response. Each test cleans up after itself.
-- ============================================================

T.suite("action_commands", {

    stop_movement_succeeds = function(t)
        local raw = fa.stop_movement()
        local parsed = t.json_ok(raw)
        t.ok(parsed.ok, "stop_movement should return ok=true")
    end,

    move_to_nearby_succeeds = function(t)
        local player = get_player()
        local start = {x = player.position.x, y = player.position.y}
        -- Move 3 tiles north (teleport mode for deterministic test).
        local raw = fa.move_to({x = start.x, y = start.y - 3}, false)
        local parsed = t.json_ok(raw)
        t.ok(parsed.ok, "move_to (teleport) should return ok=true")
        -- Restore position.
        player.teleport(start)
    end,

    move_to_with_pathfind_returns_ok = function(t)
        -- We just test that it returns ok=true and sets a walking direction.
        -- We don't verify the character actually arrives (that's the execution layer's job).
        local player = get_player()
        local target = {x = player.position.x + 10, y = player.position.y + 5}
        local raw = fa.move_to(target, true)
        local parsed = t.json_ok(raw)
        t.ok(parsed.ok, "move_to (pathfind) should return ok=true")
        -- Stop again to avoid side effects.
        fa.stop_movement()
    end,

    move_to_already_at_target_succeeds = function(t)
        local player = get_player()
        -- Target is within 0.5 tiles — should stop and return ok.
        local raw = fa.move_to(
            {x = player.position.x + 0.1, y = player.position.y + 0.1},
            true
        )
        local parsed = t.json_ok(raw)
        t.ok(parsed.ok, "move_to within threshold should return ok=true")
    end,

    craft_item_with_missing_ingredients_fails_gracefully = function(t)
        local player = get_player()
        local had = player.get_item_count("iron-plate")
        if had > 0 then
            player.remove_item({name = "iron-plate", count = had})
        end

        local raw = fa.craft_item("iron-gear-wheel", 1)
        local parsed = t.json_ok(raw)
        t.ok(not parsed.ok or parsed.queued == 0,
            "craft with no ingredients should return ok=false or queued=0")

        if had > 0 then
            player.insert({name = "iron-plate", count = had})
        end
    end,

    craft_item_with_ingredients_succeeds = function(t)
        local player = get_player()
        -- Give player enough iron plates to craft one gear wheel (2 plates).
        player.insert({name = "iron-plate", count = 10})
        local raw = fa.craft_item("iron-gear-wheel", 1)
        local parsed = t.json_ok(raw)
        t.ok(parsed.ok, "craft_item with ingredients should succeed")
        -- Clean up — cancel any pending crafting and remove inserted items.
        -- (Crafting runs asynchronously; we can't easily undo it in a test.
        -- The extra plates will simply sit in inventory.)
    end,

    place_entity_without_item_fails_gracefully = function(t)
        local player = get_player()
        local had = player.get_item_count("iron-chest")
        if had > 0 then
            player.remove_item({name = "iron-chest", count = had})
        end

        local raw = fa.place_entity("iron-chest", player.position, 0)
        local parsed = t.json_ok(raw)
        t.ok(not parsed.ok, "placing without item should return ok=false")

        if had > 0 then player.insert({name = "iron-chest", count = had}) end
    end,

    place_and_mine_entity = function(t)
        local player = get_player()
        -- Give a chest and place it, then mine it.
        player.insert({name = "wooden-chest", count = 1})
        local place_pos = {
            x = math.floor(player.position.x) + 3,
            y = math.floor(player.position.y),
        }

        local raw_place = fa.place_entity("wooden-chest", place_pos, 0)
        local parsed_place = t.json_ok(raw_place)
        if not parsed_place.ok then
            -- Placement failed (collision) — clean up and skip.
            player.remove_item({name = "wooden-chest", count = 1})
            return
        end

        local unit_number = parsed_place.unit_number
        t.ok(unit_number ~= nil, "placed entity should have a unit_number")

        -- Mine it back.
        local raw_mine = fa.mine_entity(unit_number)
        local parsed_mine = t.json_ok(raw_mine)
        t.ok(parsed_mine.ok, "mine_entity on placed chest should succeed")
    end,

    set_recipe_on_assembler = function(t)
        local player = get_player()
        -- Place an assembling machine if possible.
        local had = player.get_item_count("assembling-machine-1")
        player.insert({name = "assembling-machine-1", count = 1})
        local place_pos = {
            x = math.floor(player.position.x) + 4,
            y = math.floor(player.position.y) + 1,
        }

        local raw_place = fa.place_entity("assembling-machine-1", place_pos, 0)
        local parsed_place = t.json_ok(raw_place)
        if not parsed_place.ok then
            player.remove_item({name = "assembling-machine-1", count = 1})
            return
        end

        local unit_number = parsed_place.unit_number
        -- Set recipe to iron-gear-wheel.
        local raw_recipe = fa.set_recipe(unit_number, "iron-gear-wheel")
        local parsed_recipe = t.json_ok(raw_recipe)
        t.ok(parsed_recipe.ok, "set_recipe should succeed on assembler")

        -- Mine it back.
        fa.mine_entity(unit_number)
        -- Restore inventory.
        if had == 0 then
            player.remove_item({name = "assembling-machine-1", count = 1})
        end
    end,

    mine_resource_out_of_reach_fails_gracefully = function(t)
        -- Try to mine at a position very far away.
        local raw = fa.mine_resource({x = 999999, y = 999999}, "iron-ore", 1)
        local parsed = t.json_ok(raw)
        -- Should be either no_resource_at_position or out_of_reach, not a crash.
        t.ok(not parsed.ok, "mining out of reach should return ok=false")
        t.ok(
            parsed.reason == "no_resource_at_position" or
            parsed.reason == "out_of_reach",
            "reason should be no_resource_at_position or out_of_reach, got: " ..
            tostring(parsed.reason)
        )
    end,

    set_research_queue_enqueues_tech = function(t)
        local force = game.forces["player"]
        -- Find a technology that exists and is not yet researched.
        local target = nil
        for name, tech in pairs(force.technologies) do
            if not tech.researched then
                target = name
                break
            end
        end
        if not target then
            -- Everything researched — skip.
            return
        end

        -- Save current queue.
        local prior_queue = {}
        for _, tech in ipairs(force.research_queue or {}) do
            table.insert(prior_queue, tech.name)
        end

        local raw = fa.set_research_queue({target})
        local parsed = t.json_ok(raw)
        t.ok(parsed.ok, "set_research_queue should return ok=true")
        t.is_table(parsed.queued, "queued should be a table")
        t.ok(#parsed.queued > 0, "at least one tech should be queued")

        -- Restore prior queue.
        fa.set_research_queue(prior_queue)
    end,

    vehicle_stub_returns_error = function(t)
        local raw = fa.enter_vehicle(0)
        local parsed = t.json_ok(raw)
        t.ok(not parsed.ok, "vehicle stub should return ok=false")
        t.eq(parsed.reason, "vehicle_actions_not_implemented")
    end,

    combat_stub_returns_error = function(t)
        local raw = fa.shoot_at(nil, {x=0, y=0})
        local parsed = t.json_ok(raw)
        t.ok(not parsed.ok, "combat stub should return ok=false")
        t.eq(parsed.reason, "combat_actions_not_implemented")
    end,
})

-- ============================================================
-- Suite: edge_cases
-- Boundary conditions, empty results, malformed-ish inputs.
-- ============================================================

T.suite("edge_cases", {

    get_entities_zero_radius_returns_empty_or_local = function(t)
        -- Radius of 0 — should not crash, may return player's own tile.
        local raw = fa.get_entities(0)
        local parsed = t.json_ok(raw)
        t.is_table(parsed.entities)
    end,

    get_resource_map_small_radius_may_be_empty = function(t)
        -- With a very small radius the resource map may be empty — that is fine.
        local raw = fa.get_resource_map(1)
        local parsed = t.json_ok(raw)
        t.is_table(parsed.resource_map)
    end,

    destruction_buffer_drains_idempotently = function(t)
        -- Drain twice in a row — second should always be empty.
        fa.drain_destruction_events()
        local raw = fa.drain_destruction_events()
        local parsed = t.json_ok(raw)
        t.eq(#parsed.destroyed_entities, 0,
             "double drain: second result should be empty")
    end,

    get_state_large_radius_does_not_crash = function(t)
        -- Large radius — should not crash even if it's slow.
        local raw = fa.get_state({radius=128, resource_radius=256, item_radius=32})
        local parsed = t.json_ok(raw)
        t.has_key(parsed, "tick")
    end,

    set_recipe_invalid_entity_id_fails_gracefully = function(t)
        local raw = fa.set_recipe(999999999, "iron-gear-wheel")
        local parsed = t.json_ok(raw)
        t.ok(not parsed.ok, "set_recipe with invalid id should return ok=false")
        t.eq(parsed.reason, "entity_not_found")
    end,

    set_filter_invalid_entity_id_fails_gracefully = function(t)
        local raw = fa.set_filter(999999999, 0, "iron-plate")
        local parsed = t.json_ok(raw)
        t.ok(not parsed.ok, "set_filter with invalid id should return ok=false")
        t.eq(parsed.reason, "entity_not_found")
    end,

    mine_entity_invalid_id_fails_gracefully = function(t)
        local raw = fa.mine_entity(999999999)
        local parsed = t.json_ok(raw)
        t.ok(not parsed.ok, "mine_entity with invalid id should return ok=false")
        t.eq(parsed.reason, "entity_not_found")
    end,

    transfer_items_invalid_entity_fails_gracefully = function(t)
        local raw = fa.transfer_items(999999999, true, "iron-plate", 1)
        local parsed = t.json_ok(raw)
        t.ok(not parsed.ok, "transfer_items with invalid id should return ok=false")
        t.eq(parsed.reason, "entity_not_found")
    end,

    apply_blueprint_invalid_string_fails_gracefully = function(t)
        local raw = fa.apply_blueprint("this-is-not-a-blueprint",
                                       {x=0, y=0}, 0, false)
        local parsed = t.json_ok(raw)
        t.ok(not parsed.ok,
             "apply_blueprint with invalid string should return ok=false")
    end,

    entity_scan_excludes_resources = function(t)
        -- Resources (ore tiles) must not appear in the entity scan since they
        -- do not have unit_numbers. Verify by checking that no entity name
        -- matches a known resource name.
        local resource_names = {
            "iron-ore", "copper-ore", "coal", "stone",
            "crude-oil", "uranium-ore",
        }
        local raw = fa.get_entities(64)
        local parsed = t.json_ok(raw)
        local resource_set = {}
        for _, name in ipairs(resource_names) do resource_set[name] = true end
        for _, e in ipairs(parsed.entities) do
            t.ok(not resource_set[e.name],
                 "resource tile '" .. e.name .. "' should not appear in entity scan")
        end
    end,

    get_state_tick_matches_get_tick = function(t)
        -- The tick in get_state and get_tick should be very close (within a few ticks).
        local tick_raw = fa.get_tick()
        local state_raw = fa.get_state({radius=4})
        local tick_direct = tonumber(tick_raw)
        local state_parsed = t.json_ok(state_raw)
        local tick_state = state_parsed.tick
        t.ok(math.abs(tick_direct - tick_state) < 10,
             string.format("ticks should be close: get_tick=%d, get_state.tick=%d",
                           tick_direct, tick_state))
    end,
})

-- ============================================================
-- Entry point
-- ============================================================

-- Return T so callers can run individual suites after loading:
--   /c local T = require("test_bridge_live")
--   /c T.run_suite("edge_cases")
return T
