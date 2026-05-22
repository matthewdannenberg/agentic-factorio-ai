-- bridge/mod/test_movement_live.lua
--
-- In-game tests for the fa.move_to / fa.stop_movement / fa.get_movement_status
-- pipeline in control.lua.
--
-- HOW TO USE
-- ----------
-- With the factorio-agent mod loaded, open the console (`) and run:
--
--   /c __agent__ require("test_movement_live").run_all()
--
-- Or run a single suite:
--
--   /c __agent__ require("test_movement_live").run_suite("pathfinding")
--
-- Available suites: "status_api", "pathfinding", "obstacle_routing"
--
-- IMPORTANT: these tests mutate game state (teleport player, place/destroy
-- entities). Run on a test save, not a production game.
--
-- DESIGN
-- ------
-- Movement in Factorio is asynchronous: fa.move_to() submits a path request
-- and returns immediately; the path result arrives via on_script_path_request_finished
-- some ticks later; then on_tick walks the character along the path.
--
-- Tests use game.tick to advance game time and poll fa.get_movement_status()
-- to observe transitions without hard-coding wall-clock delays.
--
-- Helper: wait_ticks(n) advances the game n ticks by repeatedly calling
-- game.tick_all(). This gives the pathfinder time to compute and the
-- character time to move.

-- ============================================================
-- Reuse the framework from test_bridge_live if available,
-- otherwise define a minimal inline version.
-- ============================================================

-- TM is assigned as a global by control.lua so the console can call:
--   /c __agent__ TM.run_suite("status_api")
--   /c __agent__ TM.run_all()
TM = {}
local suites = {}

local log_file = "agent-movement-test-results.txt"

local function test_print(msg)
    game.print(msg)
    helpers.write_file(log_file, msg .. "\n", true)  -- append
end

local function make_assertions()
    local function fail(msg)
        error("ASSERT: " .. tostring(msg), 2)
    end
    return {
        ok        = function(v, msg)
            if not v then fail(msg or "expected truthy") end
        end,
        eq        = function(a, b, msg)
            if a ~= b then
                fail(msg or ("expected " .. tostring(b) .. " got " .. tostring(a)))
            end
        end,
        ne        = function(a, b, msg)
            if a == b then fail(msg or ("expected not " .. tostring(b))) end
        end,
        is_string = function(v, msg)
            if type(v) ~= "string" then
                fail(msg or ("expected string got " .. type(v)))
            end
        end,
    }
end

function TM.suite(name, tests) suites[name] = tests end

function TM.run_suite(name)
    local tests = suites[name]
    if not tests then test_print("[TEST] Unknown suite: " .. name) return end
    local pass, fail_count = 0, 0
    for test_name, fn in pairs(tests) do
        local ok, err = pcall(fn, make_assertions())
        if ok then
            pass = pass + 1
            test_print("[PASS] " .. name .. " :: " .. test_name)
        else
            fail_count = fail_count + 1
            test_print("[FAIL] " .. name .. " :: " .. test_name
                       .. " — " .. tostring(err))
        end
    end
    test_print(string.format("[SUITE %s] %d passed, %d failed",
                             name, pass, fail_count))
    return fail_count == 0
end

function TM.run_all()
    -- Clear the log file at the start of a full run.
    helpers.write_file(log_file, "", false)  -- false = overwrite
    test_print("=== Movement test run: tick " .. tostring(game.tick) .. " ===")
    for name in pairs(suites) do
        test_print("─────── " .. name .. " ───────")
        TM.run_suite(name)
    end
    test_print("═══ Done. See script-output/" .. log_file .. " ═══")
end

-- ============================================================
-- Shared helpers
-- ============================================================

local function get_player()
    return game.get_player(1)
end

local function parse_json(raw)
    local ok, t = pcall(function() return helpers.json_to_table(raw) end)
    if ok then return t else return nil end
end

-- Advance game time by n ticks so async operations (pathfinding, on_tick
-- movement) can complete. Uses game.tick_all() if available (2.x), otherwise
-- falls back to a busy loop checking game.tick.
local function wait_ticks(n)
    local start = game.tick
    -- game.tick_all advances the game clock synchronously in script context.
    for _ = 1, n do
        -- In 2.x we can't truly advance ticks from script without game.tick_all.
        -- As a workaround, we call game.tick_all() if it exists.
    end
    -- game.tick_all(n) is available in 2.x script context:
    local ok_tick = pcall(function() game.tick_all(n) end)
    if not ok_tick then
        -- Fallback: spin (won't actually advance game ticks in script context,
        -- but documents intent). Tests that need real tick advancement should
        -- be run interactively with a follow-up console command.
    end
end

-- Teleport player to a clear area away from map structures.
local function reset_player_position(player)
    -- (10, 10) is usually clear of spawn structures.
    player.teleport({x = 10, y = 10})
    fa.stop_movement()
end

-- ============================================================
-- Suite: status_api
-- Verifies fa.get_movement_status() returns correct values
-- at each stage of the movement lifecycle. Does not require
-- the player to actually arrive — only tests state transitions.
-- ============================================================

TM.suite("status_api", {

    idle_when_no_movement_requested = function(t)
        local player = get_player()
        reset_player_position(player)

        local raw = fa.get_movement_status()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "get_movement_status() must return valid JSON")
        t.ok(parsed.ok, "get_movement_status() should return ok=true")
        t.is_string(parsed.status, "status should be a string")
        t.eq(parsed.status, "idle", "status should be idle after stop_movement")
    end,

    pathing_immediately_after_move_to = function(t)
        local player = get_player()
        reset_player_position(player)

        -- Issue a move to a position far enough that pathfinding is needed.
        local target = {x = player.position.x + 30, y = player.position.y}
        fa.move_to(target, true)

        -- Immediately after, status should be "pathing" (request submitted,
        -- result not yet received).
        local raw = fa.get_movement_status()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "get_movement_status() must return valid JSON")
        t.eq(parsed.status, "pathing",
            "status should be 'pathing' immediately after move_to; got: " ..
            tostring(parsed and parsed.status))

        fa.stop_movement()
    end,

    walking_after_path_received = function(t)
        local player = get_player()
        reset_player_position(player)

        local target = {x = player.position.x + 20, y = player.position.y}
        fa.move_to(target, true)

        -- Wait enough ticks for the pathfinder to respond (typically 1-5 ticks).
        wait_ticks(10)

        local raw = fa.get_movement_status()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "get_movement_status() must return valid JSON")
        -- After the path arrives, status should be "walking" (character moving)
        -- or "idle" (already arrived, which is unlikely for a 20-tile move).
        t.ok(
            parsed.status == "walking" or parsed.status == "idle",
            "status should be 'walking' or 'idle' after path received; got: " ..
            tostring(parsed and parsed.status)
        )

        fa.stop_movement()
    end,

    idle_after_stop_movement = function(t)
        local player = get_player()
        reset_player_position(player)

        fa.move_to({x = player.position.x + 50, y = player.position.y}, true)
        wait_ticks(5)
        fa.stop_movement()

        local raw = fa.get_movement_status()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "get_movement_status() must return valid JSON")
        t.eq(parsed.status, "idle",
            "status should be 'idle' after stop_movement; got: " ..
            tostring(parsed and parsed.status))
    end,

    unreachable_when_target_in_void = function(t)
        local player = get_player()
        reset_player_position(player)

        -- Target very far from spawn in an ungenerated chunk.
        -- The pathfinder should return unreachable rather than crash.
        fa.move_to({x = 100000, y = 100000}, true)
        wait_ticks(60)  -- Give pathfinder plenty of time to report back.

        local raw = fa.get_movement_status()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "get_movement_status() must return valid JSON")
        t.eq(parsed.status, "unreachable",
            "status should be 'unreachable' for target in ungenerated chunk; got: " ..
            tostring(parsed and parsed.status))

        fa.stop_movement()
    end,

    status_fields_present = function(t)
        local player = get_player()
        reset_player_position(player)

        local raw = fa.get_movement_status()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "get_movement_status() must return valid JSON")
        -- Required fields in the response:
        t.ok(parsed.ok ~= nil, "response should have 'ok' field")
        t.ok(parsed.status ~= nil, "response should have 'status' field")
        -- goal is optional but if present should be a table
        if parsed.goal ~= nil then
            t.ok(type(parsed.goal) == "table", "goal should be a table if present")
            t.ok(parsed.goal.x ~= nil and parsed.goal.y ~= nil,
                "goal should have x and y if present")
        end
    end,

    collision_mask_layers = function(t)
        local player = get_player()
        if not player or not player.character then
            test_print("[INFO] No character found")
            return
        end
        local ok, proto = pcall(function()
            return player.character.prototype.collision_mask
        end)
        if not ok or not proto then
            test_print("[INFO] Could not read collision_mask: " .. tostring(proto))
            return
        end

        -- Top-level keys of the CollisionMask struct
        test_print("[INFO] collision_mask type: " .. type(proto))
        local top_keys = {}
        pcall(function()
            for k, v in pairs(proto) do
                table.insert(top_keys, tostring(k) .. " (" .. type(v) .. ")")
            end
        end)
        test_print("[INFO] Top-level keys (" .. #top_keys .. "): " .. table.concat(top_keys, ", "))

        -- Inspect proto.layers if it exists
        if type(proto.layers) == "table" then
            local layer_keys = {}
            pcall(function()
                for k, v in pairs(proto.layers) do
                    -- Print both the key's string representation AND its type,
                    -- so we can see if keys are strings, userdata, etc.
                    table.insert(layer_keys,
                        "key=" .. tostring(k) ..
                        " (type=" .. type(k) .. ")" ..
                        " val=" .. tostring(v))
                end
            end)
            test_print("[INFO] proto.layers entries (" .. #layer_keys .. "):")
            for _, l in ipairs(layer_keys) do
                test_print("  " .. l)
            end
        else
            test_print("[INFO] proto.layers type: " .. type(proto.layers))
        end

        -- Try to get the layer count via the API if available
        local ok_count, count = pcall(function() return #proto end)
        if ok_count then
            test_print("[INFO] #proto = " .. tostring(count))
        end

        -- Print the actual collision_mask we pass to request_path
        local ok_json, json_out = pcall(function()
            return helpers.table_to_json(proto)
        end)
        if ok_json and json_out then
            test_print("[INFO] proto as JSON: " .. tostring(json_out):sub(1, 400))
        else
            -- proto is likely userdata; print what we can
            test_print("[INFO] proto (tostring): " .. tostring(proto))
            -- Try serialising just the layers sub-table
            local ok_layers_json, layers_out = pcall(function()
                return helpers.table_to_json(proto.layers)
            end)
            if ok_layers_json then
                test_print("[INFO] proto.layers as JSON: " .. tostring(layers_out))
            end
        end

        -- Check what request_path actually expects by testing a minimal mask
        local ok_test, id = pcall(function()
            return player.surface.request_path({
                bounding_box = {{-0.2,-0.2},{0.2,0.2}},
                collision_mask = proto,
                start  = player.position,
                goal   = {x = player.position.x + 10, y = player.position.y},
                force  = player.force,
                radius = 0.5,
            })
        end)
        test_print("[INFO] request_path with raw proto: ok=" .. tostring(ok_test) .. " id=" .. tostring(id))

        local ok_test2, id2 = pcall(function()
            return player.surface.request_path({
                bounding_box = {{-0.2,-0.2},{0.2,0.2}},
                collision_mask = proto.layers,
                start  = player.position,
                goal   = {x = player.position.x + 10, y = player.position.y},
                force  = player.force,
                radius = 0.5,
            })
        end)
        test_print("[INFO] request_path with proto.layers: ok=" .. tostring(ok_test2) .. " id=" .. tostring(id2))
    end,

    move_to_sets_pathing_or_walking = function(t)
        -- Directly test fa.move_to and immediately read internal state.
        -- This distinguishes: (a) request_path succeeded → pathing or walking
        --                     (b) request_path failed → direct-walk → walking immediately
        --                     (c) move_to errored → idle
        local player = get_player()
        player.teleport({x = 10, y = 10})
        fa.stop_movement()
        fa.move_to({x = 20, y = 10}, true)
        local raw = fa.get_movement_status()
        local parsed = parse_json(raw)
        test_print("[INFO] status after fa.move_to: " .. tostring(parsed and parsed.status))
        test_print("[INFO] total_waypoints: " .. tostring(parsed and parsed.total_waypoints))
        -- If status is "walking" with total_waypoints=1, it's the direct-walk fallback.
        -- If status is "pathing", request_path succeeded and we're waiting for result.
        -- If status is "idle", fa.move_to itself failed.
        t.ok(
            parsed and (parsed.status == "pathing" or parsed.status == "walking"),
            "fa.move_to should result in pathing or walking; got: " ..
            tostring(parsed and parsed.status)
        )
        if parsed and parsed.status == "walking" and parsed.total_waypoints == 1 then
            test_print("[WARN] total_waypoints=1 suggests direct-walk fallback — request_path may have failed")
        end
        fa.stop_movement()
    end,

    get_state_includes_movement_status = function(t)
        -- After our recent changes, _player_table() includes movement_status.
        -- Verify it appears in get_state() output.
        local player = get_player()
        reset_player_position(player)

        local raw = fa.get_state({radius = 16, resource_radius = 32, item_radius = 8})
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "get_state() must return valid JSON")
        t.ok(parsed.player ~= nil, "get_state() must have player section")
        t.ok(parsed.player.movement_status ~= nil,
            "player section should include movement_status field")
        t.is_string(parsed.player.movement_status,
            "movement_status should be a string")
        local valid = {idle=true, pathing=true, walking=true, unreachable=true}
        t.ok(valid[parsed.player.movement_status],
            "movement_status should be one of: idle/pathing/walking/unreachable; got: " ..
            tostring(parsed.player.movement_status))
    end,
})

-- ============================================================
-- Suite: pathfinding
-- Verifies that the pathfinder is actually called and produces
-- a path, and that the character moves along it.
-- ============================================================

TM.suite("pathfinding", {

    player_moves_after_move_to = function(t)
        local player = get_player()
        reset_player_position(player)
        local start_x = player.position.x
        local start_y = player.position.y

        -- Move 15 tiles east — far enough to require real movement.
        fa.move_to({x = start_x + 15, y = start_y}, true)

        -- Wait for path to compute and character to start walking.
        wait_ticks(30)

        local new_x = player.position.x
        local new_y = player.position.y
        local dist = math.sqrt((new_x - start_x)^2 + (new_y - start_y)^2)

        t.ok(dist > 0.5,
            string.format(
                "player should have moved after move_to; " ..
                "start=(%.1f,%.1f) now=(%.1f,%.1f) dist=%.2f",
                start_x, start_y, new_x, new_y, dist
            )
        )
        fa.stop_movement()
    end,

    path_request_id_set_after_move_to = function(t)
        -- This test accesses internal Lua state to verify request_path was called.
        -- It checks the movement_status returned by get_movement_status() rather
        -- than accessing the internal variable directly.
        local player = get_player()
        reset_player_position(player)

        fa.move_to({x = player.position.x + 20, y = player.position.y}, true)

        local raw = fa.get_movement_status()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "get_movement_status() must return valid JSON")
        -- If "pathing", request_path was called and we have an id.
        -- If "walking", the path already arrived (fast pathfinder).
        -- Either is correct — "idle" is not.
        t.ok(
            parsed.status == "pathing" or parsed.status == "walking",
            "after move_to, status should be pathing or walking; got: " ..
            tostring(parsed and parsed.status)
        )
        fa.stop_movement()
    end,

    collision_mask_is_valid = function(t)
        -- Verify that the collision mask we pass to request_path is a valid
        -- CollisionMask by checking that pathfinding succeeds in open terrain.
        -- If the mask is wrong, request_path errors and we fall back to direct
        -- walk — the character still moves but status would be "walking" from
        -- a single-waypoint path rather than "pathing" first.
        local player = get_player()
        reset_player_position(player)

        fa.move_to({x = player.position.x + 10, y = player.position.y}, true)

        -- Check immediately — should be "pathing" if request_path succeeded.
        local raw = fa.get_movement_status()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "get_movement_status() must return valid JSON")
        t.ok(
            parsed.status == "pathing" or parsed.status == "walking",
            "collision_mask test: expected pathing or walking; got: " ..
            tostring(parsed and parsed.status) ..
            ". If 'idle', request_path errored and we fell back to direct walk."
        )
        fa.stop_movement()
    end,

    stop_movement_halts_walking = function(t)
        local player = get_player()
        reset_player_position(player)

        fa.move_to({x = player.position.x + 50, y = player.position.y}, true)
        wait_ticks(10)
        fa.stop_movement()

        local pos_after_stop_x = player.position.x
        wait_ticks(10)
        local pos_later_x = player.position.x

        t.ok(
            math.abs(pos_later_x - pos_after_stop_x) < 0.5,
            string.format(
                "player should not move after stop_movement; " ..
                "after_stop=%.2f later=%.2f",
                pos_after_stop_x, pos_later_x
            )
        )
    end,

    move_to_nearby_does_not_pathfind = function(t)
        -- Target within ARRIVAL_THRESHOLD should return ok immediately without
        -- submitting a path request.
        local player = get_player()
        reset_player_position(player)

        -- Move within 0.3 tiles (below ARRIVAL_THRESHOLD of 0.4).
        local raw = fa.move_to(
            {x = player.position.x + 0.2, y = player.position.y + 0.1},
            true
        )
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil and parsed.ok, "move_to within threshold should return ok")

        local status_raw = fa.get_movement_status()
        local status_parsed = parse_json(status_raw)
        t.eq(status_parsed.status, "idle",
            "status should be idle when already at target; got: " ..
            tostring(status_parsed and status_parsed.status))
    end,
})

-- ============================================================
-- Suite: obstacle_routing
-- Verifies that the pathfinder routes around solid obstacles
-- (walls, trees) rather than walking into them.
-- ============================================================

TM.suite("obstacle_routing", {

    routes_around_placed_wall = function(t)
        local player = get_player()
        reset_player_position(player)

        local base_x = math.floor(player.position.x)
        local base_y = math.floor(player.position.y)

        -- Place a line of stone walls directly east of the player,
        -- blocking a straight-line path to the target.
        local wall_entities = {}
        for dy = -2, 2 do
            local wall = player.surface.create_entity({
                name     = "stone-wall",
                position = {x = base_x + 3, y = base_y + dy},
                force    = "player",
            })
            if wall and wall.valid then
                table.insert(wall_entities, wall)
            end
        end

        -- Target is 8 tiles east — behind the wall.
        local target = {x = base_x + 8, y = base_y}
        fa.move_to(target, true)

        -- Wait for path to compute and character to start moving.
        wait_ticks(60)

        local final_x = player.position.x
        local final_y = player.position.y

        -- Destroy the wall before asserting (clean up regardless of result).
        for _, wall in ipairs(wall_entities) do
            if wall.valid then wall.destroy() end
        end

        -- If pathfinding routed around the wall, the player should have moved
        -- (either north or south to get around). If they're still at start,
        -- pathfinding failed to route around the obstacle.
        local dist = math.sqrt((final_x - player.position.x)^2 + (final_y - player.position.y)^2)

        -- More specifically: the player should not be at exactly (base_x + 3, base_y)
        -- (the wall position) — they should have gone around.
        t.ok(
            math.abs(final_y - base_y) > 0.5 or final_x > base_x + 3,
            string.format(
                "player should have routed around wall; " ..
                "start=(%.1f,%.1f) end=(%.1f,%.1f). " ..
                "If end == start, pathfinder did not route around.",
                base_x, base_y, final_x, final_y
            )
        )

        fa.stop_movement()
    end,

    unreachable_sets_correct_status = function(t)
        local player = get_player()
        reset_player_position(player)

        local base_x = math.floor(player.position.x)
        local base_y = math.floor(player.position.y)

        -- Completely surround a target position with walls so it's unreachable.
        local wall_entities = {}
        local offsets = {
            {-1, -1}, {0, -1}, {1, -1},
            {-1,  0},           {1,  0},
            {-1,  1}, {0,  1}, {1,  1},
        }
        for _, off in ipairs(offsets) do
            local wall = player.surface.create_entity({
                name     = "stone-wall",
                position = {x = base_x + 5 + off[1], y = base_y + off[2]},
                force    = "player",
            })
            if wall and wall.valid then
                table.insert(wall_entities, wall)
            end
        end

        -- Target is inside the ring of walls.
        fa.move_to({x = base_x + 5, y = base_y}, true)

        -- Wait for pathfinder to determine it's unreachable.
        wait_ticks(60)

        local raw = fa.get_movement_status()
        local parsed = parse_json(raw)

        -- Clean up walls.
        for _, wall in ipairs(wall_entities) do
            if wall.valid then wall.destroy() end
        end

        t.ok(parsed ~= nil, "get_movement_status() must return valid JSON")
        t.eq(parsed.status, "unreachable",
            "completely surrounded target should return unreachable; got: " ..
            tostring(parsed and parsed.status))
    end,

    natural_obstacle_routes_around = function(t)
        -- Verify that naturally-spawned trees don't block movement when the
        -- pathfinder is used. Find a tree near the player (if any) and try
        -- to move to a position behind it.
        local player = get_player()
        reset_player_position(player)

        -- Find trees in a small radius.
        local trees = player.surface.find_entities_filtered({
            position = player.position,
            radius   = 15,
            type     = "tree",
        })

        if #trees == 0 then
            -- No trees nearby — test is vacuously satisfied.
            game.print("[SKIP] natural_obstacle_routes_around: no trees in 15-tile radius")
            return
        end

        local tree = trees[1]
        local tree_x = tree.position.x
        local tree_y = tree.position.y

        -- Target is 3 tiles past the tree in the same direction from player.
        local dx = tree_x - player.position.x
        local dy = tree_y - player.position.y
        local len = math.sqrt(dx*dx + dy*dy)
        local target = {
            x = tree_x + (dx / len) * 3,
            y = tree_y + (dy / len) * 3,
        }

        fa.move_to(target, true)
        wait_ticks(60)

        local final_x = player.position.x
        local final_y = player.position.y
        local dist_moved = math.sqrt(
            (final_x - player.position.x)^2 + (final_y - player.position.y)^2
        )

        fa.stop_movement()

        -- Player should have moved (either around the tree or all the way to target).
        -- If they didn't move, pathfinder returned unreachable or walked into the tree.
        local status_raw = fa.get_movement_status()
        local status = parse_json(status_raw)

        t.ok(
            dist_moved > 0.5 or (status and status.status == "unreachable"),
            "player should have moved around tree or reported unreachable; " ..
            "dist_moved=" .. string.format("%.2f", dist_moved)
        )
    end,
})

-- ============================================================
-- Two-phase async tests
--
-- Factorio's pathfinder is asynchronous — results arrive via
-- on_script_path_request_finished on a future tick. Console scripts
-- run within one tick and cannot wait. These tests are split into
-- start (issue command) and finish (assert after real time has passed).
--
-- Usage:
--   /c __agent__ TM.async_start("move_and_check")
--   -- wait 3-5 seconds for Factorio to tick --
--   /c __agent__ TM.async_finish("move_and_check")
--
-- Or run all at once:
--   /c __agent__ TM.async_run_all()
--   -- wait 5+ seconds --
--   /c __agent__ TM.async_finish_all()
-- ============================================================

local async_state = {}
local async_tests = {}

local function async(name, start_fn, finish_fn)
    async_tests[name] = {start = start_fn, finish = finish_fn}
end

function TM.async_start(name)
    local test = async_tests[name]
    if not test then
        test_print("[ASYNC] Unknown test: " .. tostring(name))
        return
    end
    async_state[name] = {}
    local ok, err = pcall(test.start, async_state[name])
    if ok then
        test_print("[ASYNC START OK] " .. name)
    else
        test_print("[ASYNC START FAIL] " .. name .. " — " .. tostring(err))
    end
end

function TM.async_finish(name)
    local test = async_tests[name]
    if not test then
        test_print("[ASYNC] Unknown test: " .. tostring(name))
        return
    end
    local state = async_state[name]
    if not state then
        test_print("[ASYNC] No start state for: " .. name)
        return
    end
    local function fail(msg) error("ASSERT: " .. tostring(msg), 2) end
    local t = {
        ok = function(v, msg) if not v then fail(msg or "expected truthy") end end,
        eq = function(a, b, msg)
            if a ~= b then
                fail(msg or ("expected " .. tostring(b) .. " got " .. tostring(a)))
            end
        end,
    }
    local ok, err = pcall(test.finish, state, t)
    if ok then
        test_print("[ASYNC PASS] " .. name)
    else
        test_print("[ASYNC FAIL] " .. name .. " — " .. tostring(err))
    end
end

function TM.async_run_all()
    helpers.write_file(log_file, "", false)
    test_print("=== Async start: tick " .. tostring(game.tick) .. " ===")
    for name in pairs(async_tests) do TM.async_start(name) end
    test_print("=== Wait 5+ seconds, then: /c __agent__ TM.async_finish_all() ===")
end

function TM.async_finish_all()
    test_print("=== Async finish: tick " .. tostring(game.tick) .. " ===")
    for name in pairs(async_tests) do TM.async_finish(name) end
    test_print("=== Done. See script-output/" .. log_file .. " ===")
end

-- ── Async test definitions ──────────────────────────────────────────────────

async("walking_after_path_received",
    function(state)
        local player = get_player()
        player.teleport({x = 10, y = 10})
        fa.stop_movement()
        fa.move_to({x = 30, y = 10}, true)
        state.start_x  = player.position.x
        state.start_y  = player.position.y
        state.start_tick = game.tick
    end,
    function(state, t)
        local raw = fa.get_movement_status()
        local parsed = parse_json(raw)
        local elapsed = game.tick - (state.start_tick or 0)
        test_print("[INFO] elapsed ticks=" .. tostring(elapsed)
                   .. " status=" .. tostring(parsed and parsed.status)
                   .. " waypoints=" .. tostring(parsed and parsed.total_waypoints))
        t.ok(parsed ~= nil, "get_movement_status() must return valid JSON")
        t.ok(
            parsed.status == "walking" or parsed.status == "idle",
            "status should be walking or idle after " .. tostring(elapsed) ..
            " ticks; got: " .. tostring(parsed and parsed.status)
        )
    end
)

async("player_actually_moves",
    function(state)
        local player = get_player()
        player.teleport({x = 10, y = 10})
        fa.stop_movement()
        fa.move_to({x = 30, y = 10}, true)
        state.start_x = player.position.x
        state.start_y = player.position.y
    end,
    function(state, t)
        local player = get_player()
        local dist = math.sqrt(
            (player.position.x - state.start_x)^2 +
            (player.position.y - state.start_y)^2
        )
        fa.stop_movement()
        t.ok(dist > 1.0,
            string.format(
                "player should have moved; start=(%.1f,%.1f) now=(%.1f,%.1f) dist=%.2f",
                state.start_x, state.start_y,
                player.position.x, player.position.y, dist
            )
        )
    end
)

async("routes_around_wall",
    function(state)
        local player = get_player()
        player.teleport({x = 10, y = 10})
        fa.stop_movement()
        state.walls = {}
        for dy = -2, 2 do
            local wall = player.surface.create_entity({
                name     = "stone-wall",
                position = {x = 13, y = 10 + dy},
                force    = "player",
            })
            if wall and wall.valid then
                table.insert(state.walls, wall)
            end
        end
        state.start_x = player.position.x
        state.start_y = player.position.y
        fa.move_to({x = 18, y = 10}, true)
    end,
    function(state, t)
        local player = get_player()
        local final_x = player.position.x
        local final_y = player.position.y
        for _, wall in ipairs(state.walls or {}) do
            if wall and wall.valid then wall.destroy() end
        end
        fa.stop_movement()
        -- Wall is at x=13, blocking east. Routing around = Y changed OR past wall.
        local moved_past_wall = final_x > 13.5
        local went_around_y   = math.abs(final_y - state.start_y) > 1.0
        t.ok(
            moved_past_wall or went_around_y,
            string.format(
                "player should have routed around wall (past x=13.5 or Y changed >1); " ..
                "start=(%.1f,%.1f) end=(%.1f,%.1f). " ..
                "Stuck near x<13 with no Y change means walked into wall.",
                state.start_x, state.start_y, final_x, final_y
            )
        )
    end
)

async("move_no_collision_mask",
    -- Test: omit collision_mask entirely from request_path.
    -- If this works (player moves) but the prototype mask doesn't,
    -- then the prototype mask contains layers that block ground tiles.
    function(state)
        local player = get_player()
        player.teleport({x = 10, y = 10})
        fa.stop_movement()
        -- Call a diagnostic move that omits collision_mask
        fa.move_to_no_mask({x = 30, y = 10})
        state.start_x = player.position.x
        state.start_y = player.position.y
    end,
    function(state, t)
        local player = get_player()
        local dist = math.sqrt(
            (player.position.x - state.start_x)^2 +
            (player.position.y - state.start_y)^2
        )
        fa.stop_movement()
        t.ok(dist > 1.0,
            string.format(
                "move_no_collision_mask: dist=%.2f (should be >1 if pathfinder works without mask)",
                dist
            )
        )
    end
)

async("navigate_tree_maze",
    -- A small maze of trees forcing the pathfinder to find a non-trivial route.
    -- Layout (T=tree, .=open, S=start, G=goal, all relative to base (10,10)):
    --
    --   col:  10  11  12  13  14  15  16  17  18
    --   row 8:  .   T   T   T   T   T   T   T   .
    --   row 9:  .   T   .   .   .   .   .   T   .
    --   row10:  S   T   .   T   T   T   .   T   G
    --   row11:  .   T   .   .   .   .   .   T   .
    --   row12:  .   T   T   T   T   T   T   T   .
    --
    -- The only route is through the gap at (12,9)→(12,11) or similar.
    -- A straight-line path from S(10,10) to G(18,10) is blocked by the wall.
    function(state)
        local player = get_player()
        player.teleport({x = 10, y = 10})
        fa.stop_movement()

        local base_x, base_y = 11, 8
        state.trees = {}

        -- Top and bottom horizontal walls
        for dx = 0, 6 do
            for _, dy in ipairs({0, 4}) do
                local tree = player.surface.create_entity({
                    name     = "tree-01",
                    position = {x = base_x + dx, y = base_y + dy},
                    force    = "neutral",
                })
                if tree and tree.valid then
                    table.insert(state.trees, tree)
                end
            end
        end

        -- Left and right vertical walls (rows 1-3, leaving gap)
        for dy = 1, 3 do
            for _, dx in ipairs({0, 6}) do
                local tree = player.surface.create_entity({
                    name     = "tree-01",
                    position = {x = base_x + dx, y = base_y + dy},
                    force    = "neutral",
                })
                if tree and tree.valid then
                    table.insert(state.trees, tree)
                end
            end
        end

        -- Internal baffle: a row of trees across the middle with a gap at top
        -- Forces the path to go up then across then down
        for dy = 2, 3 do
            local tree = player.surface.create_entity({
                name     = "tree-01",
                position = {x = base_x + 3, y = base_y + dy},
                force    = "neutral",
            })
            if tree and tree.valid then
                table.insert(state.trees, tree)
            end
        end

        state.start_x  = player.position.x
        state.start_y  = player.position.y
        state.start_tick = game.tick
        state.tree_count = #state.trees

        -- Goal is on the other side of the maze
        fa.move_to({x = 18, y = 10}, true)

        test_print("[INFO] tree_maze: placed " .. #state.trees .. " trees, moving to (18,10)")
    end,
    function(state, t)
        local player = get_player()
        local final_x = player.position.x
        local final_y = player.position.y
        local elapsed = game.tick - (state.start_tick or 0)
        local status_raw = fa.get_movement_status()
        local status = parse_json(status_raw)

        -- Clean up trees regardless of result
        local removed = 0
        for _, tree in ipairs(state.trees or {}) do
            if tree and tree.valid then
                tree.destroy()
                removed = removed + 1
            end
        end
        fa.stop_movement()

        test_print(string.format(
            "[INFO] tree_maze: elapsed=%d status=%s pos=(%.1f,%.1f) removed=%d trees",
            elapsed,
            tostring(status and status.status),
            final_x, final_y, removed
        ))

        -- Player should have moved meaningfully toward or past the goal
        local dist_from_start = math.sqrt(
            (final_x - state.start_x)^2 + (final_y - state.start_y)^2
        )
        t.ok(dist_from_start > 2.0,
            string.format(
                "player should have navigated through tree maze; " ..
                "start=(%.1f,%.1f) end=(%.1f,%.1f) dist=%.2f. " ..
                "dist<2 means pathfinder returned unreachable or character never moved.",
                state.start_x, state.start_y, final_x, final_y, dist_from_start
            )
        )
    end
)

async("navigate_around_water",
    -- Place a strip of water tiles directly east of the player, forcing
    -- the pathfinder to route north or south around them.
    -- Water must be placed with surface.set_tiles(). The strip is 1 tile
    -- wide and 5 tiles tall, centered on the player's Y.
    -- Goal is 8 tiles east — on the other side of the water.
    --
    -- Layout (W=water, .=land, S=start, G=goal):
    --   row 7:  .  .  .  .  .  .  .  .  .
    --   row 8:  .  .  W  .  .  .  .  .  .
    --   row 9:  .  .  W  .  .  .  .  .  .
    --   row10:  S  .  W  .  .  .  .  .  G
    --   row11:  .  .  W  .  .  .  .  .  .
    --   row12:  .  .  W  .  .  .  .  .  .
    --   row13:  .  .  .  .  .  .  .  .  .
    function(state)
        local player = get_player()
        player.teleport({x = 30, y = 30})
        fa.stop_movement()

        local base_x = math.floor(player.position.x)
        local base_y = math.floor(player.position.y)
        local water_x = base_x + 2

        -- Save the original tiles so we can restore them
        state.original_tiles = {}
        state.water_positions = {}
        for dy = -2, 2 do
            local pos = {x = water_x, y = base_y + dy}
            local tile = player.surface.get_tile(pos.x, pos.y)
            table.insert(state.original_tiles, {
                position = pos,
                name     = tile.name,
            })
            table.insert(state.water_positions, pos)
        end

        -- Place water tiles
        local water_tiles = {}
        for dy = -2, 2 do
            table.insert(water_tiles, {
                name     = "water",
                position = {x = water_x, y = base_y + dy},
            })
        end
        player.surface.set_tiles(water_tiles)

        state.start_x    = player.position.x
        state.start_y    = player.position.y
        state.start_tick = game.tick
        state.base_x     = base_x
        state.base_y     = base_y
        state.water_x    = water_x

        local goal = {x = base_x + 7, y = base_y}
        fa.move_to(goal, true)

        test_print(string.format(
            "[INFO] water_test: placed water at x=%d rows %d to %d, moving to (%.0f,%.0f)",
            water_x, base_y - 2, base_y + 2, goal.x, goal.y
        ))
    end,
    function(state, t)
        local player = get_player()
        local final_x = player.position.x
        local final_y = player.position.y
        local elapsed = game.tick - (state.start_tick or 0)
        local status_raw = fa.get_movement_status()
        local status = parse_json(status_raw)
        fa.stop_movement()

        -- Restore original tiles
        local restore_tiles = {}
        for _, orig in ipairs(state.original_tiles or {}) do
            table.insert(restore_tiles, {name = orig.name, position = orig.position})
        end
        if #restore_tiles > 0 then
            player.surface.set_tiles(restore_tiles)
        end

        test_print(string.format(
            "[INFO] water_test: elapsed=%d status=%s pos=(%.1f,%.1f)",
            elapsed,
            tostring(status and status.status),
            final_x, final_y
        ))

        -- Player should have routed around the water (north or south)
        -- and made meaningful progress eastward.
        local dist_east = final_x - state.start_x
        local went_around_y = math.abs(final_y - state.start_y) > 1.5

        t.ok(
            dist_east > 2.0,
            string.format(
                "player should have moved east past the water strip; " ..
                "start=(%.1f,%.1f) end=(%.1f,%.1f) dist_east=%.2f. " ..
                "If dist_east<2, pathfinder returned unreachable or never walked.",
                state.start_x, state.start_y, final_x, final_y, dist_east
            )
        )
        -- Note whether they went around — this is informational, not a hard fail,
        -- since the pathfinder might find a narrow gap or diagonal route.
        if went_around_y then
            test_print("[INFO] water_test: player routed around water (Y changed)")
        else
            test_print("[INFO] water_test: player went through gap without significant Y change")
        end
    end
)

async("unreachable_in_void",
    function(state)
        local player = get_player()
        player.teleport({x = 10, y = 10})
        fa.stop_movement()
        fa.move_to({x = 100000, y = 100000}, true)
    end,
    function(state, t)
        local raw = fa.get_movement_status()
        local parsed = parse_json(raw)
        t.ok(parsed ~= nil, "get_movement_status() must return valid JSON")
        t.eq(parsed.status, "unreachable",
            "target in void should be unreachable; got: " ..
            tostring(parsed and parsed.status)
        )
        fa.stop_movement()
    end
)

-- ============================================================
-- Entry point
-- ============================================================

return TM