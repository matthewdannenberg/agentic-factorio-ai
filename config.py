# ======================================================
#   RCON
# ======================================================

RCON_HOST = "localhost"
RCON_PORT = 25575
RCON_PASSWORD = "factorio"
RCON_TIMEOUT_S = 5.0
RCON_RECONNECT_ATTEMPTS = 5
RCON_RECONNECT_BACKOFF_S = 1.0

LOCAL_SCAN_RADIUS = 32       # tiles — used for entities, ground_items
RESOURCE_SCAN_RADIUS = 128   # tiles — used for resource patches
GROUND_ITEM_SCAN_RADIUS = 16 # tiles — used for ground_items

BITERS_ENABLED = False
TICK_INTERVAL = 10           # poll every N game ticks