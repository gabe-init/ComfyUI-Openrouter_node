import assert from "node:assert/strict";
import fs from "node:fs";
import { NODE_IDS, migrateWorkflow, migrateNodeWidgets, WIDGET_NAMES, visibleInMode, modelOptions, scrubSerializedKey } from "../web/openrouter_workflow.js";

const old = JSON.parse(fs.readFileSync(new URL("../examples/chat_mode_example.json", import.meta.url)));
const before = structuredClone(old);
migrateWorkflow(old);
const node = old.nodes.find(n => n.type === "OpenRouterNode");
assert.deepEqual(node.outputs.map(o => o.name), ["Output", "image", "Stats", "Credits", "video"]);
assert.equal(old.links.find(l => l[0] === 2)[2], 2);
assert.equal(node.widgets_values[WIDGET_NAMES.indexOf("temperature")], before.nodes[0].widgets_values[7]);
assert.equal(node.widgets_values[WIDGET_NAMES.indexOf("chat_mode")], true);
assert.equal(node.widgets_values[WIDGET_NAMES.indexOf("request_type")], "chat");
const once = structuredClone(old);
migrateWorkflow(old);
assert.deepEqual(old, once, "migration must be idempotent");

// Real current-main order, including ComfyUI's seed control widget.
const currentValues = ["", "system", "prompt", "example/model", false, true, false,
    "16:9 (1344x768)", "2K", "high", 123, "randomize", .7, "auto", true, 45];
const current = {nodes: [{id: 1, type: "OpenRouterNode", widgets_values: [...currentValues], outputs:
    ["Output", "image", "Stats", "Credits"].map(name => ({name}))}], links: [[1, 1, 2, 2, 0, "STRING"]]};
migrateWorkflow(current);
assert.deepEqual(current.nodes[0].widgets_values.slice(0, currentValues.length), currentValues);
assert.equal(current.links[0][2], 2, "current Stats links must not move");
assert.equal(current.nodes[0].widgets_values[WIDGET_NAMES.indexOf("audio_encoding")], "auto");
assert.equal(current.nodes[0].widgets_values[WIDGET_NAMES.indexOf("request_type")], "chat");

// Real historical INPUT_TYPES orders from b7501e7, d481e92 and ab115ee.
for (const [tail, expected] of [
    [[true, .3, "mistral-ocr", true], {temperature: .3, pdf_engine: "mistral-ocr", chat_mode: true, image_resolution: "auto", seed: 0}],
    [[true, "2K", 123, "randomize", .4, "auto", true], {temperature: .4, chat_mode: true, image_resolution: "2K", seed: 123, control_after_generate: "randomize"}],
    [["4K", 456, "fixed", .5, "auto", false], {temperature: .5, chat_mode: false, image_resolution: "4K", seed: 456}],
]) {
    const historicalImage = {nodes: [{id: 1, type: "OpenRouterNode", widgets_values: [...currentValues.slice(0, 7), ...tail],
        outputs: ["Output", "image", "Stats", "Credits"].map(name => ({name}))}], links: []};
    migrateWorkflow(historicalImage);
    for (const [name, value] of Object.entries(expected)) {
        assert.equal(historicalImage.nodes[0].widgets_values[WIDGET_NAMES.indexOf(name)], value, `legacy ${name}`);
    }
    assert.equal(historicalImage.nodes[0].widgets_values[WIDGET_NAMES.indexOf("request_type")], "chat");
}

// A node already migrated under the pre-audio_encoding schema (cached
// openrouter_widget_names/openrouter_schema_version) must not keep using that
// stale cache once WIDGET_NAMES changes - it must be recomputed, not shifted.
{
    const v2Names = WIDGET_NAMES.filter(name => name !== "audio_encoding");
    const v2Values = v2Names.map(name => ({
        request_type: "image", service_tier: "flex", image_quality: "high",
    })[name] ?? (name === "model" ? "example/model" : name === "temperature" ? 1 : "auto"));
    const stale = {nodes: [{id: 1, type: "OpenRouterNode", widgets_values: v2Values,
        outputs: ["Output", "image", "Stats", "Credits", "video"].map(name => ({name})),
        properties: {openrouter_widget_names: v2Names, openrouter_schema_version: 2}}], links: []};
    migrateWorkflow(stale);
    const values = stale.nodes[0].widgets_values;
    assert.equal(values[WIDGET_NAMES.indexOf("request_type")], "image", "stale cache must not shift request_type");
    assert.equal(values[WIDGET_NAMES.indexOf("service_tier")], "flex", "stale cache must not shift service_tier");
    assert.equal(values[WIDGET_NAMES.indexOf("audio_encoding")], "auto", "missing field falls back to its default");
}

// configure() (recreate/undo/copy-paste) never goes through beforeConfigureGraph -
// migrateNodeWidgets must be safe to call directly on the plain info object too.
{
    const v2Names = WIDGET_NAMES.filter(name => name !== "audio_encoding");
    const v2Values = v2Names.map(name => (name === "request_type" ? "video" : name === "model" ? "x" : "auto"));
    const info = {widgets_values: v2Values, properties: {openrouter_widget_names: v2Names, openrouter_schema_version: 2}};
    migrateNodeWidgets(info);
    assert.equal(info.widgets_values[WIDGET_NAMES.indexOf("request_type")], "video", "configure() path must not shift request_type");
    assert.equal(info.widgets_values[WIDGET_NAMES.indexOf("audio_encoding")], "auto");
}

assert(visibleInMode("system_prompt", "chat"));
assert(visibleInMode("user_message_box", "chat"));
assert(!visibleInMode("system_prompt", "image"));
assert(visibleInMode("video_duration", "video"));
assert(!visibleInMode("video_duration", "video", true));
assert(visibleInMode("video_job_id", "video", true));
assert.deepEqual(modelOptions({video: [{id: "new"}]}, "video", "saved"), ["saved", "new"]);

const live = {widgets: WIDGET_NAMES.map(name => ({name, value: name === "api_key" ? "secret" : "auto"}))};
live.widgets.find(w => w.name === "control_after_generate").options = {serialize: false};
live.widgets.push({name: "Refresh Models", serialize: false, options: {serialize: false}});
const serialized = {widgets_values: live.widgets.slice(0, -1).map(w => w.value), widgets_values_named: {api_key: "secret"}};
scrubSerializedKey(live, serialized);
assert.equal(serialized.widgets_values[0], "");
assert.equal(serialized.widgets_values_named.api_key, "", "named workflow values must not leak the key");
assert.equal(live.widgets[0].value, "secret", "execution keeps the session key");
assert.deepEqual(serialized.properties.openrouter_widget_names, WIDGET_NAMES);

const saved = {nodes: [{id: 1, type: "OpenRouterNode", ...structuredClone(serialized)}]};
saved.nodes[0].widgets_values[WIDGET_NAMES.indexOf("control_after_generate")] = "randomize";
saved.nodes[0].widgets_values[WIDGET_NAMES.indexOf("temperature")] = .4;
saved.nodes[0].widgets_values_named.temperature = .6;
migrateWorkflow(saved);
assert.equal(saved.nodes[0].widgets_values[WIDGET_NAMES.indexOf("control_after_generate")], "randomize");
assert.equal(saved.nodes[0].widgets_values[WIDGET_NAMES.indexOf("temperature")], .6, "named values take precedence on modern ComfyUI");
assert.equal(saved.nodes[0].widgets_values_named.temperature, .6);

const nested = {nodes: [], definitions: {subgraphs: [structuredClone(before)]}};
nested.definitions.subgraphs[0].floatingLinks = [{origin_id: 1, origin_slot: 2}];
migrateWorkflow(nested);
assert.equal(nested.definitions.subgraphs[0].nodes[0].outputs[2].name, "Stats");
assert.equal(nested.definitions.subgraphs[0].floatingLinks[0].origin_slot, 3);

// Exercise the real extension hooks with a minimal node and a mocked catalog.
// This catches wiring and serialization bugs that pure helper tests cannot.
let controls;
const app = {registerExtension(extension) { controls = extension; }};
const listeners = new Map();
const api = {addEventListener(name, callback) { listeners.set(name, callback); }, fetchApi: async () => ({ok: true, json: async () => ({
    chat: [{id: "chat/model"},
        {id: "openai/gpt-5.4-image-2", architecture: {output_modalities: ["text", "image"]}},
        {id: "router/image-model", architecture: {output_modalities: ["image"]}},
        {id: "google/gemini-3.1-flash-image-preview", architecture: {output_modalities: ["text", "image"]}}],
    image: [{id: "image/model", supported_parameters: {resolution: {values: ["1024x1024"]}}},
        {id: "openai/gpt-5.4-image-2", supported_parameters: {aspect_ratio: {values: ["1:1", "16:9", "auto"]}}},
        {id: "google/gemini-3.1-flash-image-preview", supported_parameters: {resolution: {values: ["512", "1K", "2K", "4K"]}}}], video: [],
})})};
const controlsSource = fs.readFileSync(new URL("../web/openrouter_controls.js", import.meta.url), "utf8");
new Function("app", "api", "NODE_IDS", "migrateWorkflow", "visibleInMode", "modelOptions", "scrubSerializedKey",
    "setInterval", controlsSource.replace(/^import .*$/gm, ""))(app, api, NODE_IDS, migrateWorkflow, visibleInMode, modelOptions, scrubSerializedKey, () => {});
class FakeNode {
    constructor() {
        this.id = 42;
        this.properties = {};
        this.widgets = WIDGET_NAMES.map(name => ({name, type: "combo", value: "auto", options: {values: ["auto"]}}));
        this.widgets.find(w => w.name === "request_type").value = "chat";
        this.widgets.find(w => w.name === "model").value = "chat/model";
        this.widgets.find(w => w.name === "video_job_id").value = "";
        this.widgets.find(w => w.name === "image_resolution").options.values = ["auto", "1K", "2K", "4K"];
        this.widgets.find(w => w.name === "control_after_generate").options.serialize = false;
        this.size = [400, 400];
        this.graph = {setDirtyCanvas() {}, change() {}, getNodeById: id => String(id) === String(this.id) ? this : null};
    }
    addWidget(type, name, value, callback, options) {
        const widget = {type, name, value, callback, options};
        this.widgets.push(widget);
        return widget;
    }
    computeSize() { return [400, 400]; }
    setSize(value) { this.size = value; }
}
await controls.beforeRegisterNodeDef(FakeNode, {name: "OpenRouterNode"});
const controlled = new FakeNode();
app.graph = controlled.graph;
controlled.onNodeCreated();
controls.setup();
await new Promise(resolve => setImmediate(resolve));
const widget = name => controlled.widgets.find(w => w.name === name);
assert.equal(widget("Refresh Models").serialize, false);
assert.equal(widget("Refresh Models").options.serialize, false);
assert.equal(widget("Resume Last Video").serialize, false);
assert.equal(widget("Resume Last Video").options.serialize, false);
assert.equal(widget("Resume Last Video").options.hidden, true);
widget("request_type").value = "image";
widget("model").value = "image/model";
widget("request_type").callback();
assert.deepEqual(widget("image_resolution").options.values, ["auto", "1024x1024"]);
assert.equal(widget("system_prompt").options.hidden, true);
widget("request_type").value = "chat";
widget("request_type").callback();
assert.deepEqual(widget("image_resolution").options.values, ["auto", "1K", "2K", "4K"]);
assert.equal(widget("system_prompt").options.hidden, false);

// Chat image settings follow the counterpart Image API capabilities, too.
widget("model").value = "openai/gpt-5.4-image-2:floor";
widget("image_resolution").value = "1K";
widget("model").callback();
assert.equal(widget("image_resolution").options.hidden, true, "unsupported inherited GPT resolution must be hidden");
assert.deepEqual(widget("aspect_ratio").options.values, ["auto", "1:1", "16:9"]);
assert.equal(widget("image_quality").options.hidden, true, "native-only image controls stay hidden in chat");
widget("image_resolution").value = "2K";
widget("model").callback();
assert.equal(widget("image_resolution").options.hidden, false, "unsupported explicit resolution remains visible for correction");
widget("model").value = "google/gemini-3.1-flash-image-preview";
widget("model").callback();
assert.deepEqual(widget("image_resolution").options.values, ["auto", "512", "1K", "2K", "4K"]);
widget("request_type").value = "image";
widget("request_type").callback();
assert.deepEqual(widget("image_resolution").options.values, ["auto", "512", "1K", "2K", "4K"], "native image resolution keeps catalog values");
widget("request_type").value = "chat";
widget("request_type").callback();
widget("model").value = "router/image-model";
widget("image_resolution").value = "auto";
widget("model").callback();
assert.equal(widget("image_resolution").options.hidden, true, "an image router without image metadata cannot promise resolution controls");
assert.equal(widget("aspect_ratio").options.hidden, true);

// Accepted jobs remain recoverable without changing subsequent submissions.
const videoJob = detail => listeners.get("openrouter.video_job")({detail});
videoJob({node_id: "42", job_id: "video-job_1"});
assert.equal(controlled.properties.openrouter_last_video_job_id, "video-job_1");
assert.equal(widget("video_job_id").value, "", "receiving a job must not enable recovery");
assert.equal(widget("request_type").value, "chat");
assert.equal(widget("Resume Last Video").options.hidden, true);
videoJob({node_id: "missing", job_id: "wrong-node"});
videoJob({node_id: "42", job_id: "https://not-a-job"});
assert.equal(controlled.properties.openrouter_last_video_job_id, "video-job_1");
widget("request_type").value = "video";
widget("request_type").callback();
assert.equal(widget("Resume Last Video").options.hidden, false);
widget("Resume Last Video").callback();
assert.equal(widget("video_job_id").value, "video-job_1");
assert.equal(widget("video_mode").options.hidden, true);
widget("video_job_id").value = "";
widget("video_job_id").callback();
videoJob({node_id: 42, job_id: "video-job_2"});
assert.equal(widget("video_job_id").value, "", "a new accepted job must still require an explicit resume click");

const savedVideo = {
    widgets_values: controlled.widgets.filter(w => w.serialize !== false).map(w => w.value),
    properties: structuredClone(controlled.properties),
};
controlled.onSerialize(savedVideo);
assert.deepEqual(savedVideo.properties.openrouter_widget_names, WIDGET_NAMES, "recovery button must not shift widget serialization");
const reopened = new FakeNode();
reopened.id = 43;
reopened.onNodeCreated();
reopened.properties = structuredClone(savedVideo.properties);
WIDGET_NAMES.forEach((name, index) => { reopened.widgets.find(w => w.name === name).value = savedVideo.widgets_values[index]; });
reopened.onConfigure();
assert.equal(reopened.widgets.find(w => w.name === "video_job_id").value, "");
assert.equal(reopened.widgets.find(w => w.name === "Resume Last Video").options.hidden, false);
reopened.widgets.find(w => w.name === "Resume Last Video").callback();
assert.equal(reopened.widgets.find(w => w.name === "video_job_id").value, "video-job_2", "saved last job must survive reopen");

// Resolve hierarchical execution IDs without touching a root node with the same local ID.
const nestedVideo = new FakeNode();
nestedVideo.onNodeCreated();
app.rootGraph = {getNodeById: id => id === "5" ? {subgraph: nestedVideo.graph} : controlled};
videoJob({node_id: "5:42", job_id: "nested-job"});
assert.equal(nestedVideo.properties.openrouter_last_video_job_id, "nested-job");
assert.equal(controlled.properties.openrouter_last_video_job_id, "video-job_2");
nestedVideo.onRemoved();
videoJob({node_id: "5:42", job_id: "removed-node"});
assert.equal(nestedVideo.properties.openrouter_last_video_job_id, "nested-job");
reopened.onRemoved();
controlled.onRemoved();

let dynamic;
const dynamicSource = fs.readFileSync(new URL("../web/openrouter_dynamic_inputs.js", import.meta.url), "utf8");
new Function("app", dynamicSource.replace(/^import .*$/gm, ""))({registerExtension(extension) { dynamic = extension; }});
class DynamicNode {
    constructor() {
        this.inputs = [{name: "image_resolution", type: "STRING", link: null}, {name: "audio_data", type: "AUDIO", link: null}];
        this.widgets = [];
        this.graph = {getNodeById() { return {outputs: [{type: "IMAGE"}]}; }, setDirtyCanvas() {}};
    }
    addInput(name, type) { this.inputs.push({name, type, link: null}); }
    removeInput(index) { this.inputs.splice(index, 1); }
}
await dynamic.beforeRegisterNodeDef(DynamicNode, {name: "OpenRouterNode"});
const dynamicNode = new DynamicNode();
dynamicNode.onNodeCreated();
dynamicNode.onConnectionsChange(1, 0, true, {origin_id: 1, origin_slot: 0});
dynamicNode.onConnectionsChange(1, 1, true, {origin_id: 1, origin_slot: 0});
assert.deepEqual(dynamicNode.inputs.map(input => input.name), ["image_resolution", "audio_data", "image"]);
dynamicNode.inputs[2].link = 10;
dynamicNode.onConnectionsChange(1, 2, true, {origin_id: 1, origin_slot: 0});
assert.deepEqual(dynamicNode.inputs.map(input => input.name), ["image_resolution", "audio_data", "image_1", "image"]);
console.log("frontend workflow, mode visibility and key serialization checks passed");
