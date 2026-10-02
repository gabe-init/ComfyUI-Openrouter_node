import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { NODE_IDS, migrateWorkflow, visibleInMode, modelOptions, scrubSerializedKey } from "./openrouter_workflow.js";

let catalog = {chat: [], image: [], video: []};
let inFlight;
const nodes = new Set();
const validJobId = value => typeof value === "string" && /^[A-Za-z0-9_-]{1,256}$/.test(value);

function executionNode(id) {
    if (typeof id !== "string" && typeof id !== "number") return;
    const path = String(id).split(":");
    if (path.some(part => !part)) return;
    let graph = app.rootGraph || app.graph;
    for (let i = 0; i < path.length; i++) {
        const node = graph?.getNodeById?.(path[i]);
        if (i === path.length - 1) return nodes.has(node) ? node : undefined;
        graph = node?.subgraph;
    }
}

function rememberVideoJob({detail}) {
    if (!validJobId(detail?.job_id)) return;
    const node = executionNode(detail.node_id);
    if (!node) return;
    node.properties ||= {};
    node.properties.openrouter_last_video_job_id = detail.job_id;
    // Remember the ID without changing the next request into a recovery job.
    node.graph?.change?.();
    update(node);
}

async function refresh(force = false) {
    if (inFlight) return inFlight;
    inFlight = (async () => {
        const response = await api.fetchApi(`/openrouter/model_catalog${force ? "?refresh=1" : ""}`);
        if (!response.ok) throw new Error("Model catalog unavailable");
        catalog = await response.json();
        for (const node of nodes) update(node);
    })().catch(() => {}).finally(() => { inFlight = null; });
    return inFlight;
}

function setVisible(widget, visible) {
    if (!widget._openrouterOriginal) widget._openrouterOriginal = {
        type: widget.type, computeSize: widget.computeSize,
    };
    widget.type = visible ? widget._openrouterOriginal.type : "converted-widget";
    widget.computeSize = visible ? widget._openrouterOriginal.computeSize : () => [0, -4];
    widget.options ||= {};
    widget.options.hidden = !visible;
    widget.syncLiveVisibilityOptions?.();
    if (widget.inputEl) widget.inputEl.style.display = visible ? "" : "none";
    if (widget.element) widget.element.style.display = visible ? "" : "none";
}

function update(node) {
    const widgets = new Map((node.widgets || []).map(w => [w.name, w]));
    const mode = widgets.get("request_type")?.value || "chat";
    const resume = !!widgets.get("video_job_id")?.value?.trim();
    const model = widgets.get("model");
    if (model) model.options.values = modelOptions(catalog, mode, model.value);
    let record = (catalog[mode] || []).find(m => m.id === model?.value);
    if (!record && mode === "chat" && typeof model?.value === "string") {
        const baseId = model.value.replace(/(?::(?:floor|nitro|online))+$/, "");
        record = (catalog.chat || []).find(m => m.id === baseId);
    }
    const chatImage = mode === "chat" && record?.architecture?.output_modalities?.includes("image");
    const imageRecord = mode === "image" ? record : chatImage
        ? (catalog.image || []).find(m => m.id === record.id) : undefined;
    const parameters = imageRecord?.supported_parameters || {};
    const imageFields = {image_resolution: "resolution", image_quality: "quality", image_background: "background", aspect_ratio: "aspect_ratio"};
    for (const widget of node.widgets || []) {
        if (widget.name === "Resume Last Video") {
            setVisible(widget, mode === "video" && validJobId(node.properties?.openrouter_last_video_job_id));
            continue;
        }
        if (widget.type === "button") continue;
        if (imageFields[widget.name]) {
            widget.options ||= {};
            // Provider-specific choices must not leak into another model/mode.
            if (!widget._openrouterValues) widget._openrouterValues = [...(widget.options?.values || [])];
            widget.options.values = [...widget._openrouterValues];
            if (!widget.options.values.includes(widget.value)) widget.options.values.push(widget.value);
        }
        let visible = visibleInMode(widget.name, mode, resume);
        if ((imageRecord || chatImage) && imageFields[widget.name]) {
            const descriptor = parameters[imageFields[widget.name]];
            // Preserve explicit unsupported values for a clear validation error;
            // hide controls only when their automatic/default value is unused.
            if (!descriptor && ["auto", "1K"].includes(widget.value)) visible = false;
            if (Array.isArray(descriptor?.values)) {
                const allowed = [...new Set(["auto", ...descriptor.values.map(String)])];
                if (!allowed.includes(widget.value)) allowed.push(widget.value);
                widget.options.values = allowed;
            }
        }
        setVisible(widget, visible);
    }
    node.setSize?.([node.size[0], node.computeSize()[1]]);
    node.graph?.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "OpenRouter.Controls",
    beforeConfigureGraph(graph) { migrateWorkflow(graph); },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_IDS.has(nodeData.name)) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            nodes.add(this);
            for (const name of ["request_type", "model", "video_job_id"]) {
                const widget = this.widgets?.find(w => w.name === name);
                if (!widget) continue;
                const callback = widget.callback;
                widget.callback = (...args) => { callback?.apply(widget, args); update(this); };
            }
            const key = this.widgets?.find(w => w.name === "api_key");
            if (key?.inputEl) key.inputEl.type = "password";
            const refreshModels = this.addWidget("button", "Refresh Models", null, () => {
                refresh(true);
                // Backend refresh is asynchronous; request its completed snapshot.
                setTimeout(() => refresh(), 1500);
                setTimeout(() => refresh(), 15000);
            }, {serialize: false});
            refreshModels.serialize = false;
            const refreshCredits = this.addWidget("button", "Refresh Credits", null, async () => {
                try {
                    const response = await api.fetchApi("/openrouter/credits");
                    const data = await response.json();
                    this.title = `OpenRouter — ${data.credits}`;
                    this.graph?.setDirtyCanvas(true);
                } catch { this.title = "OpenRouter — credits unavailable"; }
            }, {serialize: false});
            refreshCredits.serialize = false;
            const resumeVideo = this.addWidget("button", "Resume Last Video", null, () => {
                const jobId = this.properties?.openrouter_last_video_job_id;
                const job = this.widgets?.find(w => w.name === "video_job_id");
                if (!validJobId(jobId) || !job) return;
                job.value = jobId;
                job.callback?.(jobId);
                this.graph?.change?.();
                update(this);
            }, {serialize: false});
            resumeVideo.serialize = false;
            update(this);
            refresh();
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            configured?.apply(this, arguments);
            update(this);
        };
        const serialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (data) {
            serialize?.apply(this, arguments);
            scrubSerializedKey(this, data);
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () { nodes.delete(this); return removed?.apply(this, arguments); };
    },
    setup() {
        api.addEventListener("openrouter.video_job", rememberVideoJob);
        setInterval(() => { if (nodes.size) refresh(); }, 15000);
    },
});
