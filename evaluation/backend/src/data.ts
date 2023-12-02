import * as fs from "fs";
import path from "path";

const OUTPUT_PATH = path.join(__dirname, '../../../output');

interface IVector {
    x: number;
    y: number;
    z: number;
}

interface ILandmark {
    index: number;
    model3dPosition: IVector;
    normalizedPosition: IVector;
    imageX: number;
    imageY: number;
}

class FrameData {

    private static readonly FRAME_NUMBER_INDEX = 0;
    private static readonly TIMESTAMP_INDEX = 1;
    private static readonly DISTANCE_INDEX = 2;
    private static readonly GAZE_VECTOR_INDEX = 3;
    private static readonly HEAD_ROTATION_INDEX = 4;
    private static readonly CENTER_INDEX = 5;
    private static readonly HEAD_POSITION_INDEX = 6;
    private static readonly LANDMARKS_INDEX = 7;


    public frameNumber: number;
    public timestamp: number;
    public distance?: number;

    public gazeVector?: IVector;
    public headRotation?: IVector;
    public headPosition?: IVector;
    public center?: IVector;
    public landmarks: ILandmark[] = [];

    constructor(content: string) {
        const parts = content.split("\n");
        if (parts.length < 2) {
            throw new Error("Content of frame is to short.");
        }

        this.frameNumber = parseInt(parts[FrameData.FRAME_NUMBER_INDEX]);
        this.timestamp = parseInt(parts[FrameData.TIMESTAMP_INDEX]);

        if (parts.length === 2) {
            return;
        }

        this.distance = parseFloat(parts[FrameData.DISTANCE_INDEX]);

        this.gazeVector = FrameData.getVectorFromString(parts[FrameData.GAZE_VECTOR_INDEX]);
        this.headPosition = FrameData.getVectorFromString(parts[FrameData.HEAD_POSITION_INDEX]);
        this.headRotation = FrameData.getVectorFromString(parts[FrameData.HEAD_ROTATION_INDEX]);
        this.center = FrameData.getVectorFromString(parts[FrameData.CENTER_INDEX]);

        this.landmarks = [];
        for (let i = FrameData.LANDMARKS_INDEX; i < parts.length; i++) {
            this.landmarks.push(FrameData.getLandmarkFromString(parts[i]));
        }
    }

    private static getLandmarkFromString(line: string) {
        const numbers = line.split(",").map(num => parseFloat(num));
        return {
            index: numbers[0],
            model3dPosition: {
                x: numbers[1],
                y: numbers[2],
                z: numbers[3]
            },
            normalizedPosition: {
                x: numbers[4],
                y: numbers[5],
                z: numbers[6]
            },
            imageX: numbers[7],
            imageY: numbers[8]
        }
    }

    private static getVectorFromString(line: string) {
        const numbers = line.split(",").map(num => parseFloat(num));
        if (numbers.length !== 3) throw new Error("Illegal State: Array is too short.");
        return {
            x: numbers[0],
            y: numbers[1],
            z: numbers[2]
        }
    }


}


interface ILabel {
    index: number;
    foundFace: boolean;
    foundFaceApproximately: boolean;
    looksOnScreen: boolean;
}

class Label {

    private index: number;
    private foundFace: boolean;
    private foundFaceApproximately: boolean;
    private looksOnScreen: boolean;

    constructor(data: ILabel) {
        this.index = data.index;
        this.foundFaceApproximately = data.foundFaceApproximately;
        this.foundFace = data.foundFace;
        this.looksOnScreen = data.looksOnScreen;
    }

    public toString() {
        return `${this.index},${this.foundFace},${this.foundFaceApproximately},${this.looksOnScreen}`
    }

    public getIndex() {
        return this.index;
    }
}


class DataSet {

    private name: string;

    private framesData: FrameData[] = [];
    private labels: Map<number, Label> = new Map();

    constructor(private directory: string) {
        this.name = path.basename(this.directory);
        this.read();
    }

    public read() {
        this.framesData = [];

        const gazeFilePath = path.join(this.directory, "result.gaze")
        const data = fs.readFileSync(gazeFilePath, "utf-8");
        const dataWithLineBreaks = data.replace(/\r\n/g, '\n')

        const frames = dataWithLineBreaks.split("\n\n");

        for (const frame of frames) {
            this.framesData.push(new FrameData(frame));
        }


        if (this.hasLabels()) {
            // TODO read labels
        }
    }

    public getName() {
        return this.name;
    }

    public hasLabels() {
        return fs.existsSync(path.join(this.directory, "labels.csv"))
    }

    public saveLabels() {
        const labelsAsArray: Label[] = Array.from(this.labels.values());
        labelsAsArray.sort((a, b) => a.getIndex() - b.getIndex());
        const csvData = labelsAsArray.join("\n");
        fs.writeFileSync('labels.csv', csvData, 'utf8');
    }

    public getNext() {
        if (this.labels.size === this.framesData.length) {
            return undefined;
        }
        for (const frame of this.framesData) {
            if (!this.labels.has(frame.frameNumber)) {
                return frame
            }
        }
        return undefined;
    }

    public addLabel(label: ILabel) {
        this.labels.set(label.index, new Label(label));
    }

    public getSize() {
        return this.framesData.length;
    }



    public getByFrameNumber(frameNumber: number) {
        for (const frameData of this.framesData) {
            if (frameData.frameNumber === frameNumber) return frameData;
        }
        return undefined;
    }

    public getDetails() {
        return {
            name: this.getName(),
            frames: this.getSize()
        }
    }

}

export class DataSetManager {

    private dataSets: DataSet[] = [];

    private static readonly instance = new DataSetManager();

    public static getInstance() {
        return DataSetManager.instance;
    }

    private constructor() {
        this.refresh();
    }

    public getListOfDatasetNames() {
        const names = [];

        for (const dataSet of this.dataSets) {
            names.push(dataSet.getDetails());
        }
        return names;
    }

    public getDetails(name: string) {
        const dataset = this.getByName(name);
        if (dataset === undefined) return undefined;
        return dataset.getDetails();
    }

    public refresh() {
        const dataSets = fs.readdirSync(OUTPUT_PATH)
            .map(file => path.join(OUTPUT_PATH, file))
            .filter(path => fs.statSync(path).isDirectory());
        this.dataSets = [];
        for (const dataSet of dataSets) {
            this.dataSets.push(new DataSet(dataSet))
        }
    }

    public getByName(name: string) {
        for (const dataSet of this.dataSets) {
            if (dataSet.getName() === name) {
                return dataSet
            }
        }
        return undefined;
    }

    public getFrame(name: string, frameNumber: number) {
        const dataSet = this.getByName(name);
        if (dataSet === undefined) return undefined;

        const frame = dataSet.getByFrameNumber(frameNumber);
        return frame;
    }


}

