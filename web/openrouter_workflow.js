// Pure workflow migrations, also exercised without a running ComfyUI frontend.
export const NODE_IDS = new Set(["OpenRouterNode", "openrouter_node"]);
export const WIDGET_NAMES = [
    "api_key", "system_prompt", "user_message_box", "model", "web_search", "cheapest", "fastest",
    "aspect_ratio", "image_resolution", "reasoning_effort", "seed", "control_after_generate",
    "temperature", "pdf_engine", "chat_mode", "request_timeout", "audio_encoding", "request_type", "service_tier",
    "image_quality", "image_background", "video_mode", "video_duration", "video_resolution",
    "video_generate_audio", "video_wait_timeout", "video_job_id",
    "tts_voice", "tts_format", "tts_instructions", "tts_speed", "tts_speakers",
];
const HISTORICAL_NAMES = ["api_key", "system_prompt", "user_message_box", "model", "web_search",
    "cheapest", "fastest", "temperature", "pdf_engine", "chat_mode"];
const COMMON_NAMES = HISTORICAL_NAMES.slice(0, 7);
const LEGACY_TAIL = ["temperature", "pdf_engine", "chat_mode"];
const DEFAULTS = ["", "You are a helpful assistant.", "Hello, how are you?", "openai/gpt-4o", false,
    true, false, "auto", "auto", "auto", 0, "fixed", 1, "auto", false, 120, "auto", "chat", "auto",
    "auto", "auto", "text_to_video", "auto", "auto", false, 900, "",
    "auto", "auto", "", 1.0, ""];
// Schema v2 (10/2026 merge) had this exact layout but no audio_encoding.
const WIDGET_NAMES_V2 = WIDGET_NAMES.filter(name => name !== "audio_encoding");
// Schema v3 added the five tts_* widgets; v4 is the current layout.
const WIDGET_NAMES_V3 = WIDGET_NAMES.filter(name => !name.startsWith("tts_"));
// Bump whenever WIDGET_NAMES changes, so a node's cached openrouter_widget_names
// (below) gets invalidated and recomputed instead of silently going stale.
const SCHEMA_VERSION = 4;

// Widget-value migration for a single node, independent of outputs/links (those
// are only reachable with graph-level context). Exported so it can run both for
// whole-graph loads (beforeConfigureGraph) AND for a single node's own configure()
// call (recreate, undo, copy/paste) - the latter never went through
// beforeConfigureGraph, so without this, those paths silently kept stale/shifted
// positional values and the live widgets built from them collapsed wrong fields.
export function migrateNodeWidgets(node, {historical = false} = {}) {
    let names = node.properties?.openrouter_schema_version === SCHEMA_VERSION
        ? node.properties?.openrouter_widget_names : undefined;
    if (!Array.isArray(names)) {
        const version = node.properties?.openrouter_schema_version;
        const old = node.widgets_values || [];
        names = historical || typeof old[7] === "number" ? HISTORICAL_NAMES
            : version === 2 ? WIDGET_NAMES_V2 : version === 3 ? WIDGET_NAMES_V3 : WIDGET_NAMES;
        // Actual released layouts: Sep 2025 image toggle, early Apr 2026
        // image_size/seed, then aspect ratio, then reasoning and timeout.
        if (!historical && typeof old[7] === "boolean") {
            names = typeof old[8] === "string"
                ? [...COMMON_NAMES, "image_generation", "image_resolution", "seed", "control_after_generate", ...LEGACY_TAIL]
                : [...COMMON_NAMES, "image_generation", ...LEGACY_TAIL];
        } else if (!historical && typeof old[8] === "number") {
            names = [...COMMON_NAMES, "image_resolution", "seed", "control_after_generate", ...LEGACY_TAIL];
        } else if (!historical && typeof old[9] === "number") {
            names = WIDGET_NAMES_V3.filter(name => name !== "reasoning_effort");
        } else if (!historical && old.length === WIDGET_NAMES_V3.length) {
            names = WIDGET_NAMES_V3;
        } else if (!historical && old.length === WIDGET_NAMES_V2.length) {
            names = WIDGET_NAMES_V2;
        }
    }
    const values = new Map(names.map((name, i) => [name, node.widgets_values?.[i]]));
    for (const [name, value] of Object.entries(node.widgets_values_named || {})) {
        if (WIDGET_NAMES.includes(name)) values.set(name, value);
    }
    node.widgets_values = WIDGET_NAMES.map((name, i) => values.get(name) ?? DEFAULTS[i]);
    if (node.widgets_values_named) {
        node.widgets_values_named = Object.fromEntries(WIDGET_NAMES.map((name, i) => [name, node.widgets_values[i]]));
    }
    node.properties ||= {};
    node.properties.openrouter_widget_names = [...WIDGET_NAMES];
    node.properties.openrouter_schema_version = SCHEMA_VERSION;
}

export function migrateWorkflow(graph) {
    for (const subgraph of graph.definitions?.subgraphs || []) migrateWorkflow(subgraph);
    for (const node of graph.nodes || []) {
        if (!NODE_IDS.has(node.type)) continue;
        const oldOutputs = node.outputs || [];
        const historical = oldOutputs.length === 3 && oldOutputs[1]?.name === "Stats";
        if (historical) {
            oldOutputs.splice(1, 0, {name: "image", type: "IMAGE", links: null});
            for (const link of [...(graph.links || []), ...(graph.floatingLinks || [])]) {
                if (Array.isArray(link) && link[1] === node.id && link[2] >= 1) link[2] += 1;
                else if (!Array.isArray(link) && link.origin_id === node.id && link.origin_slot >= 1) link.origin_slot += 1;
            }
        }
        if (oldOutputs.length === 4) oldOutputs.push({name: "video", type: "VIDEO", links: null});
        if (oldOutputs.length === 5 && oldOutputs[4]?.name === "video") {
            oldOutputs.push({name: "audio", type: "AUDIO", links: null});
        }
        oldOutputs.forEach((output, index) => { output.slot_index = index; });
        migrateNodeWidgets(node, {historical});
    }
    return graph;
}

export function visibleInMode(name, mode, resume = false) {
    if (["api_key", "request_type", "request_timeout"].includes(name)) return true;
    if (mode === "video" && resume) return ["video_job_id", "video_wait_timeout"].includes(name);
    if (mode === "audio") {
        // Model selection and the prompt stay available; everything else
        // (chat/image/video controls) has no effect on TTS requests.
        return name === "model" || name === "user_message_box" || name.startsWith("tts_");
    }
    if (name.startsWith("video_")) return mode === "video";
    if (name.startsWith("tts_")) return mode === "audio";
    if (["service_tier", "system_prompt", "web_search", "cheapest", "fastest", "reasoning_effort",
        "temperature", "pdf_engine", "chat_mode", "audio_encoding"].includes(name)) return mode === "chat";
    if (["image_quality", "image_background"].includes(name)) return mode === "image";
    if (name === "image_resolution") return mode !== "video";
    return true;
}

export function modelOptions(catalog, mode, selected) {
    const ids = (catalog[mode] || []).map(model => model.id).filter(Boolean).sort();
    // Keep unavailable saved choices visible, so refresh never silently changes
    // the model (or its price). Execution supplies the actionable validation.
    if (selected && !ids.includes(selected)) ids.unshift(selected);
    return ids;
}

export function scrubSerializedKey(node, serialized) {
    // options.serialize controls the API prompt, not workflow persistence.
    // In particular, the seed control must remain in this positional list.
    const names = node.widgets?.filter(w => w.serialize !== false).map(w => w.name) || WIDGET_NAMES;
    const index = names.indexOf("api_key");
    if (Array.isArray(serialized.widgets_values) && index >= 0) serialized.widgets_values[index] = "";
    if (serialized.widgets_values_named) serialized.widgets_values_named.api_key = "";
    serialized.properties ||= {};
    serialized.properties.openrouter_widget_names = names;
    serialized.properties.openrouter_schema_version = 2;
}
