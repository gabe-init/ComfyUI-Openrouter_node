import { app } from "../../../scripts/app.js"

const NODE_IDS = new Set(["OpenRouterNode", "openrouter_node"]);

const CHAT_ONLY_WIDGETS = [
    "system_prompt",
    "image_generation_only",
    "web_search",
    "pdf_engine",
    "chat_mode",
];

const IMAGE_ONLY_WIDGETS = [];

const CHAT_SHARED_WIDGETS = [
    "user_message_box",
    "cheapest",
    "fastest",
    "aspect_ratio",
    "image_resolution",
    "temperature",
];

const VIDEO_WIDGETS = [
    "video_mode",
    "video_prompt",
    "video_resolution",
    "video_aspect_ratio",
    "duration",
    "generate_audio",
    "poll_interval_seconds",
    "timeout_seconds",
    "provider_json",
];

const ALL_CHAT_WIDGETS = [...CHAT_ONLY_WIDGETS, ...CHAT_SHARED_WIDGETS];
const ALL_IMAGE_WIDGETS = [...IMAGE_ONLY_WIDGETS, ...CHAT_SHARED_WIDGETS];

function normalizeValues(values) {
    if (!Array.isArray(values)) {
        return [];
    }

    const ordered = [];
    const seen = new Set();
    for (const value of values) {
        const text = String(value);
        if (seen.has(text)) {
            continue;
        }
        seen.add(text);
        ordered.push(text);
    }
    return ordered;
}

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name) ?? null;
}

function setWidgetValue(node, widget, value) {
    if (!widget) {
        return false;
    }

    const nextValue = String(value);
    const currentValue = widget.value == null ? "" : String(widget.value);
    if (currentValue === nextValue) {
        return false;
    }

    widget.value = nextValue;

    const widgetIndex = node.widgets?.indexOf(widget) ?? -1;
    if (widgetIndex >= 0 && Array.isArray(node.widgets_values)) {
        node.widgets_values[widgetIndex] = nextValue;
    }

    return true;
}

function setComboValues(widget, values) {
    if (!widget?.options) {
        return false;
    }

    const nextValues = normalizeValues(values);
    const currentValues = normalizeValues(widget.options.values);
    const sameLength = nextValues.length === currentValues.length;
    const sameValues = sameLength && nextValues.every((value, index) => value === currentValues[index]);
    if (sameValues) {
        return false;
    }

    widget.options.values = nextValues;
    return true;
}

function chooseValue(currentValue, values) {
    const currentText = currentValue == null ? "" : String(currentValue);
    if (values.includes(currentText)) {
        return currentText;
    }
    return values[0] ?? "";
}

function syncWidgetVisibility(node, requestType) {
    for (const name of ALL_CHAT_WIDGETS) {
        const widget = getWidget(node, name);
        if (widget) {
            widget.hidden = requestType !== "chat";
        }
    }

    for (const name of ALL_IMAGE_WIDGETS) {
        const widget = getWidget(node, name);
        if (widget) {
            widget.hidden = requestType !== "image";
        }
    }

    for (const name of CHAT_ONLY_WIDGETS) {
        const widget = getWidget(node, name);
        if (widget) {
            widget.hidden = requestType !== "chat";
        }
    }

    for (const name of VIDEO_WIDGETS) {
        const widget = getWidget(node, name);
        if (widget) {
            widget.hidden = requestType !== "video";
        }
    }

    requestAnimationFrame(() => {
        const size = node.computeSize?.();
        if (size) {
            node.onResize?.(size);
        }
        app.graph.setDirtyCanvas(true, true);
    });
}

function syncModelList(node, requestType, chatCapabilities, imageCapabilities, videoCapabilities, allModelValues) {
    const modelWidget = getWidget(node, "model");
    if (!modelWidget) {
        return;
    }

    let filteredValues;
    if (requestType === "chat") {
        const chatModelIds = Object.keys(chatCapabilities).filter(
            (id) => !imageCapabilities[id] || !imageCapabilities[id].is_image_only
        );
        const videoIds = new Set(Object.keys(videoCapabilities));
        filteredValues = allModelValues.filter(
            (id) => chatModelIds.includes(id) && !videoIds.has(id)
        );
    } else if (requestType === "image") {
        const imageModelIds = Object.keys(imageCapabilities);
        filteredValues = allModelValues.filter((id) => imageModelIds.includes(id));
    } else if (requestType === "video") {
        const videoIds = new Set(Object.keys(videoCapabilities));
        filteredValues = allModelValues.filter((id) => videoIds.has(id));
    }

    if (!filteredValues || !filteredValues.length) {
        return;
    }

    let changed = false;
    changed = setComboValues(modelWidget, filteredValues) || changed;
    const nextModel = chooseValue(modelWidget.value, filteredValues);
    changed = setWidgetValue(node, modelWidget, nextModel) || changed;

    if (changed) {
        requestAnimationFrame(() => {
            const size = node.computeSize?.();
            if (size) {
                node.onResize?.(size);
            }
            app.graph.setDirtyCanvas(true, true);
        });
    }
}

function syncChatModelFilter(node, globals, chatCapabilities) {
    const modelWidget = getWidget(node, "model");
    const imageOnlyWidget = getWidget(node, "image_generation_only");
    if (!modelWidget || !imageOnlyWidget) {
        return;
    }

    const requestType = String(getWidget(node, "request_type")?.value ?? "chat");
    if (requestType !== "chat") {
        return;
    }

    const imageGenerationOnly = Boolean(imageOnlyWidget.value);
    const chatModelIds = Object.keys(chatCapabilities).filter(
        (id) => !chatCapabilities[id].is_image_only
    );
    const filteredValues = imageGenerationOnly
        ? chatModelIds.filter((modelId) => chatCapabilities[modelId]?.supports_image_generation)
        : chatModelIds;

    let changed = false;
    changed = setComboValues(modelWidget, filteredValues) || changed;
    const nextModelValue = chooseValue(modelWidget.value, filteredValues);
    changed = setWidgetValue(node, modelWidget, nextModelValue) || changed;

    if (changed) {
        requestAnimationFrame(() => {
            const size = node.computeSize?.();
            if (size) {
                node.onResize?.(size);
            }
            app.graph.setDirtyCanvas(true, true);
        });
    }
}

function wrapWidgetCallback(node, widgetName, callback) {
    const widget = getWidget(node, widgetName);
    if (!widget || widget.__openrouterChatWrapped) {
        return;
    }

    const originalCallback = widget.callback;
    widget.callback = (...args) => {
        const callbackResult = originalCallback?.apply(widget, args);
        requestAnimationFrame(() => {
            callback();
        });
        return callbackResult;
    };
    widget.__openrouterChatWrapped = true;
}

function configureApiKeyWidget(node) {
    const apiKeyWidget = getWidget(node, "api_key");
    if (!apiKeyWidget || apiKeyWidget.__openrouterApiKeySecured) {
        return;
    }

    apiKeyWidget.__openrouterApiKeySecured = true;

    apiKeyWidget.hidden = true;
    apiKeyWidget.serializeValue = () => "";

    let statusWidget = getWidget(node, "openrouter_api_key_status");
    if (!statusWidget) {
        statusWidget = node.addWidget(
            "text",
            "openrouter_api_key_status",
            "API key: not set",
            () => {},
            { readonly: true, serialize: false },
        );
        statusWidget.serializeValue = () => undefined;
        if (statusWidget.inputEl) {
            statusWidget.inputEl.readOnly = true;
        }
    }

    let buttonWidget = getWidget(node, "openrouter_api_key_button");
    if (!buttonWidget) {
        buttonWidget = node.addWidget(
            "button",
            "openrouter_api_key_button",
            "Set API Key",
            async () => {
                const current = apiKeyWidget.value && apiKeyWidget.value !== "" ? String(apiKeyWidget.value) : "";
                const entered = window.prompt("Enter your OpenRouter API key", current && !current.includes("...") ? current : "");
                if (entered === null) {
                    return;
                }

                const nextKey = entered.trim();
                try {
                    if (!nextKey) {
                        const response = await fetch("/openrouter/delete_api_key", { method: "POST" });
                        const data = await response.json();
                        if (!data.success) {
                            throw new Error(data.error || "Could not delete API key");
                        }
                        apiKeyWidget.value = "";
                        statusWidget.value = "API key: not set";
                    } else {
                        const response = await fetch("/openrouter/save_api_key", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ api_key: nextKey }),
                        });
                        const data = await response.json();
                        if (!data.success) {
                            throw new Error(data.error || "Could not save API key");
                        }
                        apiKeyWidget.value = data.masked || "saved";
                        statusWidget.value = `API key saved: ${data.masked || "saved"}`;
                    }

                    const size = node.computeSize?.();
                    if (size) {
                        node.onResize?.(size);
                    }
                    app.graph.setDirtyCanvas(true, true);
                } catch (error) {
                    statusWidget.value = `API key error: ${error.message}`;
                    app.graph.setDirtyCanvas(true, true);
                }
            },
            { serialize: false },
        );
        buttonWidget.serializeValue = () => undefined;
    }

    fetch("/openrouter/api_key_status")
        .then((r) => r.json())
        .then((data) => {
            if (data.saved && data.masked) {
                apiKeyWidget.value = data.masked;
                if (statusWidget) {
                    statusWidget.value = data.source === "env"
                        ? `API key loaded from environment: ${data.masked}`
                        : `API key saved locally: ${data.masked}`;
                }
            } else if (statusWidget) {
                statusWidget.value = "API key: not set";
            }
        })
        .catch(() => {});
}

app.registerExtension({
    name: "OpenRouter.ChatControls",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_IDS.has(nodeData.name)) {
            return;
        }

        const globals = {
            allModelValues: normalizeValues(nodeData.input?.required?.model?.[0]),
        };
        const chatCapabilities = nodeData.input?.required?.model?.[1]?.chat_capabilities ?? {};
        const imageCapabilities = nodeData.input?.required?.model?.[1]?.image_capabilities ?? {};
        const videoCapabilities = nodeData.input?.required?.model?.[1]?.video_capabilities ?? {};

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            syncWidgetVisibility(this, "chat");
            configureApiKeyWidget(this);

            wrapWidgetCallback(this, "request_type", () => {
                const requestType = String(getWidget(this, "request_type")?.value ?? "chat");
                syncWidgetVisibility(this, requestType);
                syncModelList(this, requestType, chatCapabilities, imageCapabilities, videoCapabilities, globals.allModelValues);
            });

            wrapWidgetCallback(this, "image_generation_only", () => {
                syncChatModelFilter(this, globals, chatCapabilities);
            });

            requestAnimationFrame(() => {
                const requestType = String(getWidget(this, "request_type")?.value ?? "chat");
                syncWidgetVisibility(this, requestType);
                syncModelList(this, requestType, chatCapabilities, imageCapabilities, videoCapabilities, globals.allModelValues);
                syncChatModelFilter(this, globals, chatCapabilities);
            });

            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            configureApiKeyWidget(this);
            requestAnimationFrame(() => {
                const requestType = String(getWidget(this, "request_type")?.value ?? "chat");
                syncWidgetVisibility(this, requestType);
                syncModelList(this, requestType, chatCapabilities, imageCapabilities, videoCapabilities, globals.allModelValues);
                syncChatModelFilter(this, globals, chatCapabilities);
            });
        };
    },
})
