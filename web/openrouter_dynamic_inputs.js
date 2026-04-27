import { app } from "../../../scripts/app.js"

const TypeSlot = {
    Input: 1,
    Output: 2,
};

const TypeSlotEvent = {
    Connect: true,
    Disconnect: false,
};

const NODE_IDS = new Set(["OpenRouterNode", "openrouter_node"]);
const PREFIX = "image";
const TYPE = "IMAGE";

app.registerExtension({
    name: 'OpenRouter.DynamicImageInputs',
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!NODE_IDS.has(nodeData.name)) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const me = onNodeCreated?.apply(this);

            this.addInput(PREFIX, TYPE);

            const slot = this.inputs[this.inputs.length - 1];
            if (slot) {
                slot.color_off = "#666";
            }

            return me;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
            const me = onConnectionsChange?.apply(this, arguments);

            if (slotType === TypeSlot.Input) {
                if (node_slot && !node_slot.name.startsWith(PREFIX)) {
                    return me;
                }

                if (link_info && event === TypeSlotEvent.Connect) {
                    const fromNode = this.graph._nodes.find(
                        (otherNode) => otherNode.id == link_info.origin_id
                    )

                    if (fromNode) {
                        const parent_link = fromNode.outputs[link_info.origin_slot];
                        if (parent_link) {
                            node_slot.type = parent_link.type;
                            node_slot.name = `${PREFIX}_`;
                        }
                    }
                } else if (event === TypeSlotEvent.Disconnect) {
                }

                let idx = 0;
                let slot_tracker = {};
                let toRemove = [];

                for(const slot of this.inputs) {
                    if (!slot.name.startsWith(PREFIX)) {
                        idx += 1;
                        continue;
                    }

                    if (slot.link === null && idx < this.inputs.length - 1) {
                        toRemove.push(idx);
                    } else if (slot.link !== null) {
                        const name = slot.name.split('_')[0];
                        let count = (slot_tracker[name] || 0) + 1;
                        slot_tracker[name] = count;
                        slot.name = `${name}_${count}`;
                    }
                    idx += 1;
                }

                toRemove.reverse();
                for(const removeIdx of toRemove) {
                    this.removeInput(removeIdx);
                }

                let lastInput = null;
                for (let i = this.inputs.length - 1; i >= 0; i--) {
                    if (this.inputs[i].name.startsWith(PREFIX)) {
                        lastInput = this.inputs[i];
                        break;
                    }
                }

                if (!lastInput || lastInput.link !== null) {
                    this.addInput(PREFIX, TYPE);
                    const newSlot = this.inputs[this.inputs.length - 1];
                    if (newSlot) {
                        newSlot.color_off = "#666";
                    }
                }

                this?.graph?.setDirtyCanvas(true);
                return me;
            }
        }

        return nodeType;
    },
})
