import { app } from "../../../scripts/app.js"

const NODE_ID = "OpenRouterVideoNode";
const AUTO_VALUE = "auto";
const ESTIMATED_COST_WIDGET = "estimated_cost";
const LEGACY_PRICING_NOTE_WIDGET = "pricing_note";

function normalizeValues(values, fallback = []) {
    if (!Array.isArray(values)) {
        return [...fallback];
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

function removeWidget(node, widgetName) {
    const widget = getWidget(node, widgetName);
    if (!widget || !Array.isArray(node.widgets)) {
        return false;
    }

    const widgetIndex = node.widgets.indexOf(widget);
    if (widgetIndex < 0) {
        return false;
    }

    widget.onRemove?.();
    node.widgets.splice(widgetIndex, 1);
    if (Array.isArray(node.widgets_values)) {
        node.widgets_values.splice(widgetIndex, 1);
    }
    return true;
}

function parsePrice(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function parseDuration(value) {
    if (value == null || value === "" || value === AUTO_VALUE) {
        return null;
    }

    const parsed = Number(value);
    if (!Number.isFinite(parsed)) {
        return null;
    }

    return Math.trunc(parsed);
}

function parseVideoSize(size) {
    if (typeof size !== "string" || !size.includes("x")) {
        return null;
    }

    const [widthText, heightText] = size.toLowerCase().split("x", 2);
    const width = Number(widthText);
    const height = Number(heightText);
    if (!Number.isFinite(width) || !Number.isFinite(height)) {
        return null;
    }

    return { width, height };
}

function gcd(a, b) {
    let left = Math.abs(a);
    let right = Math.abs(b);
    while (right !== 0) {
        const temp = right;
        right = left % right;
        left = temp;
    }
    return left || 1;
}

function simplifyRatio(width, height) {
    const divisor = gcd(width, height);
    return `${width / divisor}:${height / divisor}`;
}

function sizeMatchesResolution(size, resolution) {
    if (!resolution || resolution === AUTO_VALUE) {
        return true;
    }

    const parsed = parseVideoSize(size);
    if (!parsed) {
        return false;
    }

    const shortSide = Math.min(parsed.width, parsed.height);
    const longSide = Math.max(parsed.width, parsed.height);
    const normalized = String(resolution).toLowerCase();

    if (normalized === "480p") {
        return shortSide === 480;
    }
    if (normalized === "720p") {
        return shortSide === 720;
    }
    if (normalized === "1080p") {
        return shortSide === 1080;
    }
    if (normalized === "4k") {
        return longSide === 3840 || longSide === 4096 || shortSide === 2160;
    }
    return true;
}

function sizeMatchesAspectRatio(size, aspectRatio) {
    if (!aspectRatio || aspectRatio === AUTO_VALUE) {
        return true;
    }

    const parsed = parseVideoSize(size);
    if (!parsed) {
        return false;
    }

    return simplifyRatio(parsed.width, parsed.height) === String(aspectRatio);
}

function matchingSupportedSizes(modelCapabilities, resolution, aspectRatio) {
    const sizes = normalizeValues(modelCapabilities.supported_sizes);
    return sizes.filter((size) => sizeMatchesResolution(size, resolution) && sizeMatchesAspectRatio(size, aspectRatio));
}

function resolutionSuffixes(resolution) {
    if (!resolution || resolution === AUTO_VALUE) {
        return [];
    }

    const normalized = String(resolution).toLowerCase();
    const mapping = {
        "480p": ["480p"],
        "720p": ["720p"],
        "1080p": ["1080p"],
        "1k": ["1k", "1024p"],
        "2k": ["2k", "2048p"],
        "4k": ["4k", "2160p"],
    };
    return mapping[normalized] ?? [normalized];
}

function formatCostDisplay(minCost, maxCost) {
    if (minCost == null || maxCost == null) {
        return "N/A";
    }
    if (Math.abs(minCost - maxCost) < 1e-9) {
        return `$${minCost.toFixed(4)}`;
    }
    return `$${minCost.toFixed(4)}-$${maxCost.toFixed(4)}`;
}

function estimateTokenPricedCost(modelCapabilities, settings) {
    const duration = settings.duration;
    if (!Number.isFinite(duration)) {
        return null;
    }

    const pricingSkus = modelCapabilities.pricing_skus ?? {};
    const generateAudio = settings.generateAudio !== false;
    const skuKey = !generateAudio && pricingSkus.video_tokens_without_audio ? "video_tokens_without_audio" : "video_tokens";
    const rate = parsePrice(pricingSkus[skuKey]);
    if (rate == null) {
        return null;
    }

    let sizes = matchingSupportedSizes(modelCapabilities, settings.resolution, settings.aspectRatio);
    if (!sizes.length) {
        sizes = normalizeValues(modelCapabilities.supported_sizes);
    }
    if (!sizes.length) {
        return null;
    }

    const costs = [];
    for (const size of sizes) {
        const parsed = parseVideoSize(size);
        if (!parsed) {
            continue;
        }

        const tokenCount = (parsed.width * parsed.height * duration * 24) / 1024;
        costs.push(tokenCount * rate);
    }

    if (!costs.length) {
        return null;
    }

    return {
        display: formatCostDisplay(Math.min(...costs), Math.max(...costs)),
        note: costs.length > 1
            ? "Estimation sur plusieurs tailles publiees. Choisis ratio + resolution pour affiner."
            : "Estimation basee sur la taille publiee correspondant aux selections.",
    };
}

function estimateModeResolutionCost(modelCapabilities, settings) {
    const duration = settings.duration;
    if (!Number.isFinite(duration)) {
        return null;
    }

    const pricingSkus = modelCapabilities.pricing_skus ?? {};
    const prefix = settings.mode === "text_to_video" ? "text_to_video" : "image_to_video";
    const candidateResolutions = settings.resolution && settings.resolution !== AUTO_VALUE
        ? [settings.resolution]
        : normalizeValues(modelCapabilities.supported_resolutions);

    const costs = [];
    for (const resolution of candidateResolutions) {
        for (const suffix of resolutionSuffixes(resolution)) {
            const skuKey = `${prefix}_duration_seconds_${suffix}`;
            const rate = parsePrice(pricingSkus[skuKey]);
            if (rate == null) {
                continue;
            }
            costs.push(duration * rate);
            break;
        }
    }

    if (!costs.length) {
        return null;
    }

    return {
        display: formatCostDisplay(Math.min(...costs), Math.max(...costs)),
        note: costs.length > 1
            ? "Estimation sur plusieurs resolutions possibles pour ce mode."
            : "Estimation basee sur le mode et la resolution selectionnes.",
    };
}

function estimateDurationCost(modelCapabilities, settings) {
    const duration = settings.duration;
    if (!Number.isFinite(duration)) {
        return null;
    }

    const pricingSkus = modelCapabilities.pricing_skus ?? {};
    const generateAudio = settings.generateAudio !== false;
    const candidateResolutions = settings.resolution && settings.resolution !== AUTO_VALUE
        ? [settings.resolution]
        : normalizeValues(modelCapabilities.supported_resolutions).length
            ? normalizeValues(modelCapabilities.supported_resolutions)
            : [null];

    const costs = [];
    for (const resolution of candidateResolutions) {
        const suffixes = resolutionSuffixes(resolution);
        const keyCandidates = [];

        if (generateAudio) {
            keyCandidates.push(...suffixes.map((suffix) => `duration_seconds_with_audio_${suffix}`));
            keyCandidates.push("duration_seconds_with_audio");
        } else {
            keyCandidates.push(...suffixes.map((suffix) => `duration_seconds_without_audio_${suffix}`));
            keyCandidates.push("duration_seconds_without_audio");
        }

        keyCandidates.push(...suffixes.map((suffix) => `duration_seconds_${suffix}`));
        keyCandidates.push("duration_seconds");

        for (const skuKey of keyCandidates) {
            const rate = parsePrice(pricingSkus[skuKey]);
            if (rate == null) {
                continue;
            }
            costs.push(duration * rate);
            break;
        }
    }

    if (!costs.length) {
        return null;
    }

    return {
        display: formatCostDisplay(Math.min(...costs), Math.max(...costs)),
        note: costs.length > 1
            ? "Estimation sur plusieurs resolutions publiees pour ce modele."
            : "Estimation basee sur les SKUs publics du modele.",
    };
}

function estimateVideoCost(modelCapabilities, settings) {
    const pricingSkus = modelCapabilities.pricing_skus ?? {};
    const pricingKeys = Object.keys(pricingSkus);
    if (!pricingKeys.length) {
        return {
            display: "N/A",
            note: "Tarification publique indisponible pour ce modele.",
        };
    }

    if (pricingKeys.some((key) => key.startsWith("text_to_video_duration_seconds_") || key.startsWith("image_to_video_duration_seconds_"))) {
        return estimateModeResolutionCost(modelCapabilities, settings)
            ?? {
                display: "N/A",
                note: "Selection insuffisante pour calculer le cout. Renseigne surtout la duree et la resolution.",
            };
    }

    if (pricingKeys.includes("video_tokens") || pricingKeys.includes("video_tokens_without_audio")) {
        return estimateTokenPricedCost(modelCapabilities, settings)
            ?? {
                display: "N/A",
                note: "Selection insuffisante pour calculer le cout. Renseigne surtout la duree, la resolution et le ratio.",
            };
    }

    return estimateDurationCost(modelCapabilities, settings)
        ?? {
            display: "N/A",
            note: "Selection insuffisante pour calculer le cout. Renseigne surtout la duree.",
        };
}

function ensureInfoWidgets(node) {
    removeWidget(node, LEGACY_PRICING_NOTE_WIDGET);

    let estimatedCostWidget = getWidget(node, ESTIMATED_COST_WIDGET);
    if (!estimatedCostWidget) {
        estimatedCostWidget = node.addWidget(
            "text",
            ESTIMATED_COST_WIDGET,
            "N/A",
            () => {},
            { readonly: true, serialize: false },
        );
        estimatedCostWidget.serializeValue = () => undefined;
        if (estimatedCostWidget.inputEl) {
            estimatedCostWidget.inputEl.readOnly = true;
            estimatedCostWidget.inputEl.style.opacity = "0.85";
        }
    }
    return { estimatedCostWidget };
}

function configureAdvancedProviderWidget(node) {
    const providerWidget = getWidget(node, "provider_json");
    if (!providerWidget) {
        return false;
    }

    if (providerWidget.inputEl) {
        providerWidget.inputEl.placeholder = "Optional provider options JSON (advanced)";
        providerWidget.inputEl.title = "Optional provider routing/options JSON for advanced OpenRouter video settings.";
        providerWidget.inputEl.style.opacity = "0.92";
    }

    const currentValue = providerWidget.value == null ? "" : String(providerWidget.value).trim();
    if (currentValue === "{}") {
        return setWidgetValue(node, providerWidget, "");
    }

    return false;
}

function getCurrentSettings(node) {
    return {
        mode: getWidget(node, "mode")?.value ?? "text_to_video",
        resolution: getWidget(node, "resolution")?.value ?? AUTO_VALUE,
        aspectRatio: getWidget(node, "aspect_ratio")?.value ?? AUTO_VALUE,
        duration: parseDuration(getWidget(node, "duration")?.value),
        generateAudio: Boolean(getWidget(node, "generate_audio")?.value),
    };
}

function syncEstimatedCost(node, capabilities) {
    const modelWidget = getWidget(node, "model");
    const { estimatedCostWidget } = ensureInfoWidgets(node);
    if (!modelWidget || !estimatedCostWidget) {
        return false;
    }

    const modelId = modelWidget.value == null ? "" : String(modelWidget.value);
    const modelCapabilities = capabilities[modelId] ?? {};
    const settings = getCurrentSettings(node);
    const estimate = estimateVideoCost(modelCapabilities, settings);

    return setWidgetValue(node, estimatedCostWidget, estimate?.display ?? "N/A");
}

function wrapWidgetCallback(node, widgetName, callback) {
    const widget = getWidget(node, widgetName);
    if (!widget || widget.__openrouterVideoWrapped) {
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
    widget.__openrouterVideoWrapped = true;
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
    if (
        widgetIndex >= 0
        && Array.isArray(node.widgets_values)
        && widget.options?.serialize !== false
    ) {
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

function chooseValue(currentValue, values, fallbackValue) {
    const currentText = currentValue == null ? "" : String(currentValue);
    if (values.includes(currentText)) {
        return currentText;
    }
    if (fallbackValue != null && values.includes(String(fallbackValue))) {
        return String(fallbackValue);
    }
    return values[0] ?? "";
}

function syncVideoWidgets(node, globals, capabilities, options = {}) {
    const modelWidget = getWidget(node, "model");
    if (!modelWidget) {
        return;
    }

    const modelId = modelWidget.value == null ? "" : String(modelWidget.value);
    const modelCapabilities = capabilities[modelId] ?? {};

    const modeWidget = getWidget(node, "mode");
    const resolutionWidget = getWidget(node, "resolution");
    const aspectRatioWidget = getWidget(node, "aspect_ratio");
    const durationWidget = getWidget(node, "duration");

    const modeValues = normalizeValues(
        modelCapabilities.supported_modes?.length ? modelCapabilities.supported_modes : globals.modeValues,
        globals.modeValues,
    );
    const resolutionValues = normalizeValues(
        modelCapabilities.supported_resolutions?.length
            ? [AUTO_VALUE, ...modelCapabilities.supported_resolutions]
            : globals.resolutionValues,
        globals.resolutionValues,
    );
    const aspectRatioValues = normalizeValues(
        modelCapabilities.supported_aspect_ratios?.length
            ? [AUTO_VALUE, ...modelCapabilities.supported_aspect_ratios]
            : globals.aspectRatioValues,
        globals.aspectRatioValues,
    );
    const durationValues = normalizeValues(
        modelCapabilities.supported_durations?.length
            ? [AUTO_VALUE, ...modelCapabilities.supported_durations]
            : globals.durationValues,
        globals.durationValues,
    );

    let changed = false;
    changed = setComboValues(modeWidget, modeValues) || changed;
    changed = setComboValues(resolutionWidget, resolutionValues) || changed;
    changed = setComboValues(aspectRatioWidget, aspectRatioValues) || changed;
    changed = setComboValues(durationWidget, durationValues) || changed;

    if (modeWidget) {
        const nextMode = chooseValue(modeWidget.value, modeValues, modeValues[0]);
        changed = setWidgetValue(node, modeWidget, nextMode) || changed;
    }

    if (resolutionWidget) {
        const nextResolution = chooseValue(resolutionWidget.value, resolutionValues, AUTO_VALUE);
        changed = setWidgetValue(node, resolutionWidget, nextResolution) || changed;
    }

    if (aspectRatioWidget) {
        const nextAspectRatio = chooseValue(aspectRatioWidget.value, aspectRatioValues, AUTO_VALUE);
        changed = setWidgetValue(node, aspectRatioWidget, nextAspectRatio) || changed;
    }

    if (durationWidget) {
        const minimumDuration = modelCapabilities.minimum_duration;
        const shouldForceMinimum = options.forceMinimumDuration === true && minimumDuration;
        let nextDuration = chooseValue(durationWidget.value, durationValues, minimumDuration ?? AUTO_VALUE);

        if (shouldForceMinimum) {
            nextDuration = String(minimumDuration);
        } else {
            const currentDuration = durationWidget.value == null ? "" : String(durationWidget.value);
            const currentIsAutomatic = currentDuration === "" || currentDuration === AUTO_VALUE;
            if (currentIsAutomatic && minimumDuration && durationValues.includes(String(minimumDuration))) {
                nextDuration = String(minimumDuration);
            }
        }

        changed = setWidgetValue(node, durationWidget, nextDuration) || changed;
    }

    changed = syncEstimatedCost(node, capabilities) || changed;

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

app.registerExtension({
    name: "OpenRouter.VideoControls",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) {
            return;
        }

        const globals = {
            modeValues: normalizeValues(nodeData.input?.required?.mode?.[0]),
            resolutionValues: normalizeValues(nodeData.input?.required?.resolution?.[0]),
            aspectRatioValues: normalizeValues(nodeData.input?.required?.aspect_ratio?.[0]),
            durationValues: normalizeValues(nodeData.input?.required?.duration?.[0]),
        };
        const capabilities = nodeData.input?.required?.model?.[1]?.video_capabilities ?? {};

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            configureAdvancedProviderWidget(this);
            ensureInfoWidgets(this);
            wrapWidgetCallback(this, "model", () => {
                syncVideoWidgets(this, globals, capabilities, { forceMinimumDuration: true });
            });
            for (const widgetName of ["mode", "resolution", "aspect_ratio", "duration", "generate_audio"]) {
                wrapWidgetCallback(this, widgetName, () => {
                    syncVideoWidgets(this, globals, capabilities);
                });
            }

            requestAnimationFrame(() => {
                syncVideoWidgets(this, globals, capabilities);
            });

            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            requestAnimationFrame(() => {
                configureAdvancedProviderWidget(this);
                syncVideoWidgets(this, globals, capabilities);
            });
        };
    },
})
