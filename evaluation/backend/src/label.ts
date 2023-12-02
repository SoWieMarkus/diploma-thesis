import fs from "fs";
import { getSubDirectories, getFilesWithExtension } from "./io/helper";
import path from "path";

const BASE_PATH = path.join(__dirname, '../../../../output');

class LabelSet {

    public static readonly FALSE = 0;
    public static readonly TRUE = 1;
    public static readonly UNKNOWN = 2;
    public static readonly NOT_SET = 3;

    private labels: Map<number, number> = new Map();
    private labels2: Map<number, number> = new Map();
    private keys: number[] = [];

    constructor(private video: string, private face?: string) {
        const images = getFilesWithExtension(this.getImagesFolder(), ".jpg");

        for (let image of images) {
            const frameNumber = parseInt(image.name.replace(".jpg", ""));
            this.labels.set(frameNumber, LabelSet.NOT_SET);
            this.labels2.set(frameNumber, LabelSet.NOT_SET);
        }

        const outputFile = this.getOutputFile()
        if (fs.existsSync(outputFile)) {
            const content = fs.readFileSync(outputFile, 'utf-8');
            const lines = content.split("\n");

            lines.shift();

            for (let line of lines) {
                const parts = line.split(";");
                this.set(parseInt(parts[0]), parseInt(parts[1]))
            }

        }

        this.keys = Array.from(this.labels.keys());
        this.keys.sort((a, b) => a - b)
    }

    public getByFrameNumber(frameNumber: number) {
        return this.labels.get(frameNumber);
    }

    public first() {
        return {
            frame: this.keys[0],
            progress: 0
        };
    }

    public getFirstMissing() {
        for (let i = 0; i < this.keys.length; i++) {
            if (this.labels.get(this.keys[i]) === LabelSet.NOT_SET)
                return {
                    frame: this.keys[i],
                    progress: i
                };
        }
        return this.first();
    }

    public next(frameNumber: number) {
        for (let i = 0; i < this.keys.length; i++) {
            if (this.keys[i] > frameNumber) {
                return {
                    frame: this.keys[i],
                    progress: i
                };
            }
        }
    }

    public has(frameNumber: number) {
        return this.labels.has(frameNumber);
    }

    public check(video: string, face?: string) {
        return video === this.video && this.face === face;
    }

    public getSize() {
        return this.labels.size;
    }

    public getImagesFolder() {
        if (this.face !== undefined && this.face !== "blink") {
            return path.join(BASE_PATH, this.video, "face", this.face, "images");
        }
        return path.join(BASE_PATH, this.video, "images")
    }

    public getOutputFile() {
        if (this.face === "blink") {
            return path.join(BASE_PATH, this.video, "blink.labels");
        }
        if (this.face !== undefined) {
            return path.join(BASE_PATH, this.video, "face", this.face, `${this.face}.labels`);
        }
        return path.join(BASE_PATH, this.video, "base.labels");
    }

    public getOutputFile2() {
        if (this.face !== undefined) {
            return path.join(BASE_PATH, this.video, "face", this.face, `${this.face}_2.labels`);
        }
        return path.join(BASE_PATH, this.video, "base_2.labels");
    }

    public set(frame: number, value: number) {
        if (!this.labels.has(frame)) return false;
        this.labels.set(frame, value)
        return true;
    }

    public isComplete() {
        for (let value of this.labels.values()) {
            if (value === LabelSet.NOT_SET) return false;
        }
        return true;
    }

    public save() {

        const outputFile = this.getOutputFile();
        const exportContent: string[] = []
        exportContent.push(`${this.isComplete()}`)
        for (let key of this.keys) {
            exportContent.push(`${key};${this.labels.get(key)}`)
        }
        const contentString = exportContent.join("\n");

        fs.writeFileSync(outputFile, contentString)
    }

    public previous(frameNumber: number) {
        for (let i = this.keys.length; i >= 0; i--) {
            if (this.keys[i] < frameNumber) return {
                frame: this.keys[i],
                progress: i
            };
        }
        return this.first();
    }

    public getMissing() {
        let count = 0;
        for (let value of this.labels.values()) {
            if (value === LabelSet.NOT_SET) {
                count++;
            }
        }
        return count;
    }

    public getRedirect() {
        if (this.face === undefined) return `label/base/${this.video}`;
        if (this.face === "blink") return `label/blink/${this.video}`;
        return `label/face/${this.video}/${this.face}`
    }

    public getDetails() {
        const video = this.video;
        const face = this.face;

        const redirect = this.getRedirect();

        const size = this.getSize();
        const missing = this.getMissing();
        const done = size - missing;

        return {
            redirect: redirect,
            video: video,
            face: face,
            size: size,
            missing: missing,
            done: done
        }
    }

}

export class LabelManager {

    private static readonly instance = new LabelManager();

    private labelSets: LabelSet[] = [];

    public static getInstance() {
        return LabelManager.instance;
    }

    private constructor() {
        const videoPaths = getSubDirectories(BASE_PATH);

        for (let video of videoPaths) {
            this.labelSets.push(new LabelSet(video.name, undefined));
            this.labelSets.push(new LabelSet(video.name, "blink"));
            const facePaths = getSubDirectories(path.join(video.path, "face"));
            for (let face of facePaths) {
                this.labelSets.push(new LabelSet(video.name, face.name));
            }
        }
    }

    public getLabelset(video: string, face?: string) {
        for (let set of this.labelSets) {
            if (set.check(video, face)) return set;
        }
        return undefined;
    }

    public getDetails() {
        const details = []
        for (let labelset of this.labelSets) {
            details.push(labelset.getDetails());
        }

        let total = 0;
        let totalMissing = 0;
        let totalDone = 0;
        for (let detail of details) {
            total += detail.size;
            totalMissing += detail.missing;
            totalDone += detail.done;
        }

        return {
            details: {
                total: total,
                missing: totalMissing,
                done: totalDone
            },
            links: details
        }
    }
}