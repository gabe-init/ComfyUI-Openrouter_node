import { app } from "../../../scripts/app.js"

const NODE_IDS = new Set(["OpenRouterNode", "openrouter_node"]);

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

function syncChatModelFilter(node, globals, capabilities) {
    const modelWidget = getWidget(node, "model");
    const imageOnlyWidget = getWidget(node, "image_generation_only");
    if (!modelWidget || !imageOnlyWidget) {
        return;
    }

    const imageGenerationOnly = Boolean(imageOnlyWidget.value);
    const filteredValues = imageGenerationOnly
        ? globals.allModelValues.filter((modelId) => capabilities[modelId]?.supports_image_generation)
        : globals.allModelValues;

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

app.registerExtension({
    name: "OpenRouter.ChatControls",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_IDS.has(nodeData.name)) {
            return;
        }

        const globals = {
            allModelValues: normalizeValues(nodeData.input?.required?.model?.[0]),
        };
        const capabilities = nodeData.input?.required?.model?.[1]?.chat_capabilities ?? {};

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            wrapWidgetCallback(this, "image_generation_only", () => {
                syncChatModelFilter(this, globals, capabilities);
            });

            requestAnimationFrame(() => {
                syncChatModelFilter(this, globals, capabilities);
            });

            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            requestAnimationFrame(() => {
                syncChatModelFilter(this, globals, capabilities);
            });
        };
    },
})
