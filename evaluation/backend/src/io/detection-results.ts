import fs from "fs";

export abstract class DetectionResult {

    private static readonly INDEX_FRAME_NUMBER = 0;
    private static readonly INDEX_TIMESTAMP = 1;

    public frameNumber: number;
    public timestamp: number;

    constructor(content: string[]) {

        if (content.length < 2) {
            throw new Error("Content of frame is to short.");
        }

        this.frameNumber = parseInt(content[DetectionResult.INDEX_FRAME_NUMBER]);
        this.timestamp = parseInt(content[DetectionResult.INDEX_TIMESTAMP]);

    }

}

export abstract class DetectionResults<I extends DetectionResult> {

    private map: Map<number, I> = new Map();

    private completed = false;
    private size = 0;

    constructor(private modelName: string, private path: string) {
        this.read()
    }

    public getSize() {
        return this.size;
    }

    private read() {
        this.map.clear();

        const data = fs.readFileSync(this.path, "utf-8");
        const dataWithLineBreaks = data.replace(/\r\n/g, '\n')
        const frames = dataWithLineBreaks.split("\n\n");

        const isCompleted = frames.shift();
        this.completed = isCompleted === "$True";

        this.size = frames.length;

        for (const frame of frames) {
            const faceDetectionResult = this.parse(frame.split("\n"));
            const frameNumber = faceDetectionResult.frameNumber;
            if (this.map.has(frameNumber)) throw new Error("Illegal State");
            this.map.set(frameNumber, faceDetectionResult);
        }
    }

    abstract parse(content: string[]): I;

    public isCompleted() {
        return this.completed;
    }

    public getModelName() {
        return this.modelName;
    }

    public getByFrameNumber(frameNumber: number) {
        return this.map.get(frameNumber);
    }

}

