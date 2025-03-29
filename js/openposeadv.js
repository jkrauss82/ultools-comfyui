import { app } from "/scripts/app.js";
import "./fabric.min.js";

const connect_keypoints = [
	[0, 1],   [1, 2],  [2, 3],   [3, 4],
	[1, 5],   [5, 6],  [6, 7],   [1, 8],
	[8, 9],   [9, 10], [1, 11],  [11, 12],
	[12, 13], [14, 0], [14, 16], [15, 0],
	[15, 17]
]

const connect_color = [
	[  0,   0, 255],
	[255,   0,   0],
	[255, 170,   0],
	[255, 255,   0],
	[255,  85,   0],
	[170, 255,   0],
	[ 85, 255,   0],
	[  0, 255,   0],

	[  0, 255,  85],
	[  0, 255, 170],
	[  0, 255, 255],
	[  0, 170, 255],
	[  0,  85, 255],
	[ 85,   0, 255],

	[170,   0, 255],
	[255,   0, 255],
	[255,   0, 170],
	[255,   0,  85]
]

const DEFAULT_KEYPOINTS = [
  [241,  77], [241, 120], [191, 118], [177, 183],
  [163, 252], [298, 118], [317, 182], [332, 245],
  [225, 241], [213, 359], [215, 454], [270, 240],
  [282, 360], [286, 456], [232,  59], [253,  60],
  [225,  70], [260,  72]
]

async function readFileToText(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = async () => {
            resolve(reader.result);
        };
        reader.onerror = async () => {
            reject(reader.error);
        }
        reader.readAsText(file);
    })
}

export async function loadImageAsync(imageURL) {
    return new Promise((resolve) => {
        const e = new Image();
        e.setAttribute('crossorigin', 'anonymous');
        e.addEventListener("load", () => { resolve(e); });
        e.src = imageURL;
        return e;
    });
}

async function canvasToBlob(canvas) {
    return new Promise(function(resolve) {
        canvas.toBlob(resolve);
    });
}

export class OpenPosePanel {
    node = null
	canvas = null
	canvasElem = null
	panel = null

	undo_history = []
	redo_history = []

	visibleEyes = true
	flipped = false
	lockMode = false

	constructor(panel, node) {
		this.panel = panel
        this.node = node

        const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0)
        const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0)

        // set wh according to current ui state
        const mh = document.getElementById("comfyui-body-top").offsetHeight || 0
        const mw = document.getElementById("comfyui-body-left").offsetWidth || 0

        const width = vw ? vw - mw : 1024
        const height = vh ? vh - mh : 768
        this.panel.style.width = `${width}px`
        this.panel.style.height = `${height}px`
        this.panel.style.left = `${mw}px`
        this.panel.style.top = `${mh}px`
        this.panel.style.marginTop = '0'
        this.panel.style.marginLeft = '0'
        this.panel.style.zIndex = 1000

		const rootHtml = `
<canvas class="openpose-editor-canvas" />
<div class="canvas-drag-overlay" />
<input bind:this={fileInput} class="openpose-file-input" type="file" accept=".json" />
`;
		const container = this.panel.addHTML(rootHtml, "openpose-container");
        container.style.width = "100%";
        container.style.height = "100%";
        container.style.margin = "auto";
        container.style.display = "flex";

        const dragOverlay = container.querySelector(".canvas-drag-overlay")
		dragOverlay.style.pointerEvents = "none";
		dragOverlay.style.visibility = "hidden";
		dragOverlay.style.display = "flex";
		dragOverlay.style.alignItems = "center";
		dragOverlay.style.justifyContent = "center";
		dragOverlay.style.width = "100%";
		dragOverlay.style.height = "100%";
		dragOverlay.style.color = "white";
		dragOverlay.style.fontSize = "2.5em";
		dragOverlay.style.fontFamily = "inherit";
		dragOverlay.style.fontWeight = "600";
		dragOverlay.style.lineHeight = "100%";
		dragOverlay.style.background = "rgba(0,0,0,0.5)";
		dragOverlay.style.margin = "0.25rem";
		dragOverlay.style.borderRadius = "0.25rem";
		dragOverlay.style.border = "0.5px solid";
		dragOverlay.style.position = "absolute";

        this.canvasWidth = 512;
        this.canvasHeight = 512;

		this.canvasElem = container.querySelector(".openpose-editor-canvas")
		this.canvasElem.width = this.canvasWidth
		this.canvasElem.height = this.canvasHeight
		this.canvasElem.style.margin = "0.25rem"
		this.canvasElem.style.borderRadius = "0.25rem"
		this.canvasElem.style.border = "0.5px solid"

		this.canvas = this.initCanvas(this.canvasElem)

        this.fileInput = container.querySelector(".openpose-file-input");
        this.fileInput.style.display = "none";
        this.fileInput.addEventListener("change", this.onLoad.bind(this))

		this.panel.addButton("Add", () => {
            this.addPose()
            this.saveToNode()
            this.updateOutputs()
        })
		this.panel.addButton("Remove", () => {
            this.removePose()
            this.saveToNode()
            this.updateOutputs()
        })
		this.panel.addButton("Reset", () => {
            this.resetCanvas()
            this.saveToNode()
            this.updateOutputs()
        })
		this.panel.addButton("Save", () => this.save())
		this.panel.addButton("Load", () => this.load())

		const widthLabel = document.createElement("label")
		widthLabel.innerHTML = "Width"
		widthLabel.style.fontFamily = "Arial"
		widthLabel.style.padding = "0 0.5rem";
		widthLabel.style.color = "#ccc";
		this.widthInput = document.createElement("input")
		this.widthInput.style.background = "#1c1c1c";
		this.widthInput.style.color = "#aaa";
		this.widthInput.setAttribute("type", "number")
		this.widthInput.setAttribute("min", "64")
		this.widthInput.setAttribute("max", "4096")
		this.widthInput.setAttribute("step", "64")
		this.widthInput.setAttribute("type", "number")
		this.widthInput.addEventListener("change", () => {
			this.resizeCanvas(+this.widthInput.value, +this.heightInput.value);
			this.saveToNode();
		})

		const heightLabel = document.createElement("label")
		heightLabel.innerHTML = "Height"
		heightLabel.style.fontFamily = "Arial"
		heightLabel.style.padding = "0 0.5rem";
		heightLabel.style.color = "#aaa";
		this.heightInput = document.createElement("input")
		this.heightInput.style.background = "#1c1c1c";
		this.heightInput.style.color = "#ccc";
		this.heightInput.setAttribute("type", "number")
		this.heightInput.setAttribute("min", "64")
		this.heightInput.setAttribute("max", "4096")
		this.heightInput.setAttribute("step", "64")
		this.heightInput.addEventListener("change", () => {
			this.resizeCanvas(+this.widthInput.value, +this.heightInput.value);
			this.saveToNode();
		})

		this.panel.footer.appendChild(widthLabel);
		this.panel.footer.appendChild(this.widthInput);
		this.panel.footer.appendChild(heightLabel);
		this.panel.footer.appendChild(this.heightInput);

        this.panel.addButton("Close", () => this.panel.close())

        this.panel.footer.style.textAlign = 'center'
        const buttons = this.panel.footer.querySelectorAll('button')
        for (let i = 0; i < buttons.length; i++) {
            buttons[i].style.marginLeft = '5px'
        }

        if (this.node.properties.savedPose) {
            const error = this.loadJSON(this.node.properties.savedPose);
            if (error) {
                console.error("[OpenPose Editor] Failed to restore saved pose JSON", error)
				this.resizeCanvas(this.canvasWidth, this.canvasHeight)
                this.setPose(DEFAULT_KEYPOINTS)
            }
            this.undo_history.push(JSON.stringify(this.canvas));
        }
        else {
			this.resizeCanvas(this.canvasWidth, this.canvasHeight)
            this.setPose(DEFAULT_KEYPOINTS)
        }

		const keyHandler = this.onKeyDown.bind(this);

		document.addEventListener("keydown", keyHandler)
		this.panel.onClose = () => {
			document.removeEventListener("keydown", keyHandler)
		}
	}

	onKeyDown(e) {
        let triggered = false
		if (e.key === "z" && e.ctrlKey) {
			this.undo()
            triggered = true
		}
		else if (e.key === "y" && e.ctrlKey) {
			this.redo()
            triggered = true
		}
		else if (e.keyCode == 8 || e.keyCode == 46) {
			this.removePose()
            triggered = true
		}
		else if (e.keyCode == 27) {
			this.panel.close()
            triggered = true
		}
        if (triggered) {
            e.preventDefault()
            e.stopImmediatePropagation()
        }
	}

	addPose(keypoints = undefined) {
		if (keypoints === undefined){
			keypoints = DEFAULT_KEYPOINTS;
		}

		const group = new fabric.Group([], {
            subTargetCheck: true,
            interactive: true
        })

		function makeCircle(color, left, top, line1, line2, line3, line4, line5) {
			var c = new fabric.Circle({
				left: left,
				top: top,
				strokeWidth: 1,
				radius: 5,
				fill: color,
				stroke: color,
				originX: 'center',
				originY: 'center',
			});
			c.hasControls = c.hasBorders = false;

			c.line1 = line1;
			c.line2 = line2;
			c.line3 = line3;
			c.line4 = line4;
			c.line5 = line5;

			return c;
		}

		function makeLine(coords, color) {
			return new fabric.Line(coords, {
				fill: color,
				stroke: color,
				strokeWidth: 10,
				selectable: false,
				evented: false,
				originX: 'center',
				originY: 'center',
			});
		}

		const lines = []
		const circles = []

		for (let i = 0; i < connect_keypoints.length; i++){
			// Specify the idx to be connected. If it is [0, 1], connect 0 and 1.
			const item = connect_keypoints[i]
            if (keypoints[item[0]] && keypoints[item[1]]) {
                const line = makeLine(keypoints[item[0]].concat(keypoints[item[1]]), `rgba(${connect_color[i].join(", ")}, 0.7)`)
                lines.push(line)
                this.canvas.add(line)
                line['id'] = item[0]
            }
		}

        for (let i = 0; i < keypoints.length; i++){
            // const list = connect_keypoints.filter(item => item.includes(i));
            const list = []
            connect_keypoints.filter((item, idx) => {
            	if(item.includes(i)){
            		list.push(lines[idx])
            		return idx
            	}
            })
            const circle = makeCircle(`rgb(${connect_color[i].join(", ")})`, keypoints[i][0], keypoints[i][1], ...list)
            circle["id"] = i
            circles.push(circle)
            // this.canvas.add(circle)
            group.add(circle);
        }

        group.lines = lines
        group.circles = circles

        this.canvas.discardActiveObject();
        this.canvas.setActiveObject(group);
        this.canvas.add(group);
        // console.warn(group)
        // this.canvas.setActiveObject(group)
        // group.toActiveSelection();
        this.canvas.requestRenderAll();
    }

    setPose(keypoints){
        this.canvas.clear()
        this.canvas.backgroundColor = "#000"

        const res = [];
        for (let i = 0; i < keypoints.length; i += 18) {
            const chunk = keypoints.slice(i, i + 18);
            res.push(chunk);
        }

        for (const item of res) {
            this.addPose(item)
            this.canvas.discardActiveObject()
        }

        this.updateOutputs()
        this.saveToNode()
    }

    calcResolution(width, height){
        const viewportWidth = window.innerWidth / 2.25;
        const viewportHeight = window.innerHeight * 0.75;
        const ratio = Math.min(viewportWidth / width, viewportHeight / height);
        return {width: width * ratio, height: height * ratio}
    }

    resizeCanvas(width, height){
        let resolution = this.calcResolution(width, height)

        this.canvasWidth = width;
        this.canvasHeight = height;

		this.widthInput.value = `${width}`
		this.heightInput.value = `${height}`

        this.canvas.setWidth(width);
        this.canvas.setHeight(height);
        this.canvasElem.style.width = resolution["width"] + "px"
        this.canvasElem.style.height = resolution["height"] + "px"
        this.canvasElem.nextElementSibling.style.width = resolution["width"] + "px"
        this.canvasElem.nextElementSibling.style.height = resolution["height"] + "px"
        this.canvasElem.parentElement.style.width = resolution["width"] + "px"
        this.canvasElem.parentElement.style.height = resolution["height"] + "px"
        this.canvasElem.parentElement.style.margin = "auto";
    }

    undo() {
        if (this.undo_history.length > 0) {
            this.lockMode = true
            if (this.undo_history.length > 1) {
                this.redo_history.push(this.undo_history.pop())
            }
            const content = this.undo_history[this.undo_history.length - 1]
            this.canvas.loadFromJSON(content, () => {
                this.canvas.renderAll()
                this.lockMode = false
            })
        }
    }

    redo() {
        if (this.redo_history.length > 0) {
            this.lockMode = true;
            const content = this.redo_history.pop();
            this.undo_history.push(content);
            this.canvas.loadFromJSON(content, () => {
                this.canvas.renderAll();
                this.lockMode = false;
            });
        }
    }

    initCanvas(elem){
        const canvas = new fabric.Canvas(elem, {
            backgroundColor: '#000',
            // selection: false,
            preserveObjectStacking: true
        });

        this.undo_history = [];
        this.redo_history = [];

        const updateLines = (target) => {
            if ("_objects" in target) {
                const flipX = target.flipX ? -1 : 1;
                const flipY = target.flipY ? -1 : 1;
                this.flipped = flipX * flipY === -1;
                const showEyes = this.flipped ? !this.visibleEyes : this.visibleEyes;

                if (target.angle === 0) {
                    const rtop = target.top
                    const rleft = target.left
                    for (const item of target._objects){
                        let p = item;
                        p.scaleX = 1;
                        p.scaleY = 1;
                        const top = rtop + p.top * target.scaleY * flipY + target.height * target.scaleY / 2;
                        const left = rleft + p.left * target.scaleX * flipX + (target.width * target.scaleX / 2);
                        p['_top'] = top;
                        p['_left'] = left;
                        if (p["id"] === 0) {
                            p.line1 && p.line1.set({ 'x1': left, 'y1': top });
                        }else{
                            p.line1 && p.line1.set({ 'x2': left, 'y2': top });
                        }
                        if (p['id'] === 14 || p['id'] === 15) {
                            p.radius = showEyes ? 5 : 0.3;
                            if (p.line1) p.line1.strokeWidth = showEyes ? 10 : 0;
                            if (p.line2) p.line2.strokeWidth = showEyes ? 10 : 0;
                        }
                        p.line2 && p.line2.set({ 'x1': left, 'y1': top });
                        p.line3 && p.line3.set({ 'x1': left, 'y1': top });
                        p.line4 && p.line4.set({ 'x1': left, 'y1': top });
                        p.line5 && p.line5.set({ 'x1': left, 'y1': top });

                    }
                } else {
                    const aCoords = target.aCoords;
                    const center = {'x': (aCoords.tl.x + aCoords.br.x)/2, 'y': (aCoords.tl.y + aCoords.br.y)/2};
                    const rad = target.angle * Math.PI / 180;
                    const sin = Math.sin(rad);
                    const cos = Math.cos(rad);

                    for (const item of target._objects){
                        let p = item;
                        const p_top = p.top * target.scaleY * flipY;
                        const p_left = p.left * target.scaleX * flipX;
                        const left = center.x + p_left * cos - p_top * sin;
                        const top = center.y + p_left * sin + p_top * cos;
                        p['_top'] = top;
                        p['_left'] = left;
                        if (p["id"] === 0) {
                            p.line1 && p.line1.set({ 'x1': left, 'y1': top });
                        }else{
                            p.line1 && p.line1.set({ 'x2': left, 'y2': top });
                        }
                        if (p['id'] === 14 || p['id'] === 15) {
                            p.radius = showEyes ? 5 : 0.3;
                            if (p.line1) p.line1.strokeWidth = showEyes ? 10 : 0;
                            if (p.line2) p.line2.strokeWidth = showEyes ? 10 : 0;
                        }
                        p.line2 && p.line2.set({ 'x1': left, 'y1': top });
                        p.line3 && p.line3.set({ 'x1': left, 'y1': top });
                        p.line4 && p.line4.set({ 'x1': left, 'y1': top });
                        p.line5 && p.line5.set({ 'x1': left, 'y1': top });
                    }
                }
                target.setCoords();
            } else {
                const p = target;
                const group = p.group;

                const flipX = group.flipX ? -1 : 1;
                const flipY = group.flipY ? -1 : 1;
                this.flipped = flipX * flipY === -1;
                const showEyes = this.flipped ? !this.visibleEyes : this.visibleEyes;

                const aCoords = group.aCoords;
                const center = {'x': (aCoords.tl.x + aCoords.br.x)/2, 'y': (aCoords.tl.y + aCoords.br.y)/2};
                const rad = target.angle * Math.PI / 180;
                const sin = Math.sin(rad);
                const cos = Math.cos(rad);

                const p_top = p.top * group.scaleY * flipY;
                const p_left = p.left * group.scaleX * flipX;
                const left = center.x + p_left * cos - p_top * sin;
                const top = center.y + p_left * sin + p_top * cos;

                if (p["id"] === 0) {
                    p.line1 && p.line1.set({ 'x1': left, 'y1': top });
                }else{
                    p.line1 && p.line1.set({ 'x2': left, 'y2': top });
                }
                p.line2 && p.line2.set({ 'x1': left, 'y1': top });
                p.line3 && p.line3.set({ 'x1': left, 'y1': top });
                p.line4 && p.line4.set({ 'x1': left, 'y1': top });
                p.line5 && p.line5.set({ 'x1': left, 'y1': top });

                group.setCoords();
            }
            canvas.renderAll();
        }

        canvas.on('object:moving', (e) => {
            updateLines(e.target);
        });

        canvas.on('object:scaling', (e) => {
            updateLines(e.target);
            canvas.renderAll();
        });

        canvas.on('object:rotating', (e) => {
            updateLines(e.target);
            canvas.renderAll();
        });

        canvas.on("object:modified", () => {
            if (this.lockMode) return;
            this.undo_history.push(JSON.stringify(canvas));
            this.redo_history.length = 0;
            this.saveToNode()
        })

        return canvas;
    }

    saveToNode() {
        this.node.setProperty("savedPose", this.serializeJSON());
        this.uploadCanvasAsFile()
    }

    async captureCanvasClean() {
        this.lockMode = true;

        this.canvas.getObjects("image").forEach((img) => {
            img.opacity = 0;
        })
        if (this.canvas.backgroundImage)
            this.canvas.backgroundImage.opacity = 0
        this.canvas.discardActiveObject();
        this.canvas.renderAll()

        const blob = await canvasToBlob(this.canvasElem);

        this.canvas.getObjects("image").forEach((img) => {
            img.opacity = 1;
        })
        if (this.canvas.backgroundImage)
            this.canvas.backgroundImage.opacity = 0.5
        this.canvas.renderAll()

        this.lockMode = false;

        return blob
    }

    async uploadCanvasAsFile() {
		try {
            const blob = await this.captureCanvasClean()
            const filename = `ComfyUI_OpenPose_${this.node.id}.png`;

			const body = new FormData();
			body.append("image", blob, filename);
			body.append("overwrite", "true");

			const resp = await fetch("/upload/image", {
				method: "POST",
				body,
			});

			if (resp.status === 200) {
				const data = await resp.json();
                await this.node.setImage(data.name)
			} else {
                console.error('OpenPoseAdv: error in uploadCanvasAsFile', resp.status + " - " + resp.statusText)
				alert(resp.status + " - " + resp.statusText);
			}
		} catch (error) {
            console.error(error)
			alert(error);
		}
    }

    removePose() {
        const selection = this.canvas.getActiveObject();
        if (!selection || !("lines" in selection))
            return

        this.undo_history.push(JSON.stringify(this.canvas))

        for (const line of selection.lines){
            this.canvas.remove(line)
        }

        this.canvas.remove(selection)
    }

    resetCanvas() {
        this.canvas.clear()
        this.canvas.backgroundColor = "#000"
    }

    load() {
        this.fileInput.value = null;
        this.fileInput.click();
    }

    async onLoad(e) {
        const file = this.fileInput.files[0];
        const text = await readFileToText(file);
        const error = await this.loadJSON(text);
        if (error != null) {
            app.ui.dialog.show(error);
        }
        else {
            this.saveToNode();
        }
    }

    serializeJSON() {
        const groups = this.canvas.getObjects().filter(i => i.type === "group");
        const keypoints = groups.map(g => {
            const circles = g.getObjects().filter(i => i.type === "circle");
            return circles.map(c => [c.oCoords.tl.x, c.oCoords.tl.y]);
        })

        const json = JSON.stringify({
            "width": this.canvas.width,
            "height": this.canvas.height,
            "keypoints": keypoints
        }, null, 4)
        return json;
    }

    save() {
        const json = this.serializeJSON()
        const blob = new Blob([json], {
            type: "application/json"
        });
        const filename = "pose-" + Date.now().toString() + ".json"
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = filename;
        a.click();
        URL.revokeObjectURL(a.href);
    }

    loadJSON(text) {
        let json = JSON.parse(text)
        // convert controlnet aux format to ours
        if (Array.isArray(json) && json.length > 0 && json[0].people && Array.isArray(json[0].people) && json[0].people.length > 0 && json[0].people[0].pose_keypoints_2d && Array.isArray(json[0].people[0].pose_keypoints_2d) && json[0].people[0].pose_keypoints_2d.length > 0 && json[0].canvas_height && json[0].canvas_width) {
            console.log('converting controlnet aux format')
            const temp = { height: json[0].canvas_height, width: json[0].canvas_width, keypoints: [] }
            for (const dude of json[0].people) {
                if (!dude.pose_keypoints_2d) continue
                const pose = dude.pose_keypoints_2d
                const converted = []
                // for (let i = 0; i < pose.length; i = i + 3) {
                let idx = 0
                for (let i = 0; i < DEFAULT_KEYPOINTS.length; i++) {
                    try {
                        const isRelativeValues = pose[idx] <= 1 && post[idx] >= 0 && pose[idx+1] <= 1 && post[idx+1] >= 0
                        if (pose[idx]) {
                            const convx = isRelativeValues ? pose[idx] * temp.width : pose[idx]
                            const convy = isRelativeValues ? pose[idx+1] * temp.height : pose[idx+1]
                            converted.push([ convx, convy ])
                        } else {
                            converted.push([ 0, 0 ])
                            console.warn(`no keypoint observed for idx ${idx}`)
                        }
                    } catch (err) {
                        console.warn('error while trying to convert from controlnet aux input', err)
                    }
                    idx = idx + 3
                }
                temp.keypoints.push(converted)
            }
            json = temp
        }
        // console.log(json)
        if (json["width"] && json["height"]) {
            this.resizeCanvas(json["width"], json["height"])
        } else {
            return 'width, height is invalid';
        }
        this.resetCanvas()
        const keypoints = json["keypoints"] || []
        for (const group of keypoints) {
            if (group.length % 18 === 0) {
                this.addPose(group)
            } else {
                return 'keypoints is invalid'
            }
        }
        this.updateOutputs()
        return null
    }

    updateOutputs() {
        const groups = this.canvas.getObjects().filter(i => i.type === "group")
        if (groups.length == this.node.outputs.length -1) return

        while (groups.length > this.node.outputs.length -1) {
            this.node.addOutput(`MASK_POSE_${this.node.outputs.length}`, 'MASK')
        }
        while (groups.length < this.node.outputs.length -1) {
            this.node.removeOutput(this.node.outputs.length -1)
        }
    }
}
