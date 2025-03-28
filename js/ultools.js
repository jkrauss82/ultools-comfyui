import { app } from "../../scripts/app.js"
import './exif-reader.js'
import "./fabric.min.js"
import { OpenPosePanel, loadImageAsync } from "./openposeadv.js"

app.registerExtension({

	name: "ultools",

	async setup(app) {
		async function getImgExifData(webpFile) {
			const reader = new FileReader()
			reader.readAsArrayBuffer(webpFile)

			return new Promise((resolve, reject) => {
				reader.onloadend = function() {
					const buffer = reader.result
					const view = new DataView(buffer)
					let offset = 0

					// Search for the "EXIF" tag
					while (offset < view.byteLength - 4) {
						if (view.getUint32(offset, true) === 0x46495845 /* "EXIF" in big-endian */) {
							const exifOffset = offset + 6
							const exifData = buffer.slice(exifOffset)
							const exifString = new TextDecoder().decode(exifData).replaceAll(String.fromCharCode(0), '') //Remove Null Terminators from string
							let exifJsonString = exifString.slice(exifString.indexOf("Workflow")) //find beginning of Workflow Exif Tag
							let promptregex = "(?<!\{)}Prompt:{(?![\w\s]*[\}])" //Regex to split }Prompt:{ // Hacky as fuck - theoretically if somebody has a text encode with dynamic prompts turned off, they could enter }Prompt:{ which breaks this
							let exifJsonStringMap = new Map([
								["workflow",exifJsonString.slice(9,exifJsonString.search(promptregex)+1)], // Remove "Workflow:" keyword in front of the JSON workflow data passed
								["prompt",exifJsonString.substring((exifJsonString.search(promptregex)+8))] //Find and remove "Prompt:" keyword in front of the JSON prompt data
							])
							let fullJson = Object.fromEntries(exifJsonStringMap) //object to pass back

							resolve(fullJson)
						}

						offset++
					}

					reject(new Error('EXIF metadata not found'))
				}
			})
		}

		const handleFile = app.handleFile;

		app.handleFile = async function(file) { // Add the 'file' parameter to the function definition
			// TODO: use ExifReader for webp as well
			if (file.type === "image/webp") {
				const webpInfo = await getImgExifData(file)
				if (webpInfo) {
					if (webpInfo.workflow) {
						this.loadGraphData(JSON.parse(webpInfo.workflow))
					} else if (webpInfo.prompt) {
						this.loadApiJson(JSON.parse(webpInfo.prompt))
					}
				}
			}
			else if (file.type === "image/jpeg") {
				const tags = await ExifReader.load(file)
				// read workflow from ImageDescription
				if (tags) {
					try {
						if (tags['ImageDescription'] && tags['ImageDescription'].description) {
							const workflow = JSON.parse(tags['ImageDescription'].description)
							this.loadGraphData(workflow)
						} else if (tags['Make'] && tags['Make'].description) {
							const prompt = JSON.parse(tags['Make'].description)
							this.loadApiJson(prompt)
						}
					} catch (err) {
						console.warn('Error getting workflow from image', tags['ImageDescription'])
						return handleFile.apply(this, arguments)
					}
				}
			}
			else {
				return handleFile.apply(this, arguments)
			}
		}
	},

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name !== "OpenPoseEditorAdv") {
			return
		}

		fabric.Object.prototype.transparentCorners = false;
		fabric.Object.prototype.cornerColor = '#108ce6';
		fabric.Object.prototype.borderColor = '#108ce6';
		fabric.Object.prototype.cornerSize = 10;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

			if (!this.properties) {
				this.properties = {};
				this.properties.savedPose = "";
			}

			this.serialize_widgets = true;

			// Output & widget
			this.imageWidget = this.widgets.find(w => w.name === "image");
			this.imageWidget.callback = this.showImage.bind(this);
			this.imageWidget.disabled = true
			// console.error(this);

			// Non-serialized widgets
			//this.jsonWidget = this.addWidget("text", "", this.properties.savedPose, "savedPose");
			this.jsonWidget = this.widgets.find(w => w.name === "savedPose");
			this.jsonWidget.disabled = true
			this.jsonWidget.serialize = true
			this.jsonWidget.hidden = true

			this.openWidget = this.addWidget("button", "open editor", "image", () => {
				const graphCanvas = LiteGraph.LGraphCanvas.active_canvas
				if (graphCanvas == null)
					return;

				const panel = graphCanvas.createPanel("OpenPose Editor", { closable: true });
				panel.node = this;
				panel.classList.add("openpose-editor");

				this.openPosePanel = new OpenPosePanel(panel, this);
				document.body.appendChild(this.openPosePanel.panel);
			});
			this.openWidget.serialize = false;

			while (this.outputs.length > 1) {
				this.removeOutput(this.outputs.length -1)
			}

			// On load if we have a value then render the image
			// The value isnt set immediately so we need to wait a moment
			// No change callbacks seem to be fired on initial setting of the value
			requestAnimationFrame(async () => {
				if (this.imageWidget.value) {
					await this.setImage(this.imageWidget.value);
				}
			});
		}

		nodeType.prototype.showImage = async function(name) {
			let folder_separator = name.lastIndexOf("/");
			let subfolder = "";
			if (folder_separator > -1) {
				subfolder = name.substring(0, folder_separator);
				name = name.substring(folder_separator + 1);
			}
			const img = await loadImageAsync(`/view?filename=${name}&type=input&subfolder=${subfolder}&t=${Date.now()}`);
			this.imgs = [img];
			app.graph.setDirtyCanvas(true);
		}

		nodeType.prototype.setImage = async function(name) {
			this.imageWidget.value = name;
			await this.showImage(name);
		}

		const onPropertyChanged = nodeType.prototype.onPropertyChanged;
		nodeType.prototype.onPropertyChanged = function(property, value) {
			if (property === "savedPose") {
				this.jsonWidget.value = value;
			}
			else {
				if(onPropertyChanged)
					onPropertyChanged.apply(this, arguments)
			}
		}
	}
})
