import { IVector, getVectorFromString } from "./helper";
import fs from "fs";
import { DetectionResults, DetectionResult } from "./detection-results";

interface ILandmark {
    index: number;
    model3dPosition: IVector;
    normalizedPosition: IVector;
    imageX: number;
    imageY: number;
}

export class FaceDetectionResult extends DetectionResult {

    private static readonly INDEX_DISTANCE = 2;
    private static readonly INDEX_ROTATION = 3;
    private static readonly INDEX_CENTER = 4;
    private static readonly INDEX_POSITION = 5;
    private static readonly INDEX_BLINK_SCORE = 6;
    private static readonly INDEX_LANDMARKS = 7;
    public landmarks: ILandmark[] = [];

    public distance?: number;
    public center?: IVector;
    public rotationVector?: IVector;
    public position?: IVector;
    public blinkScore?: number;

    constructor(content: string[]) {
        super(content);

        if (content.length === 2) {
            return;
        }

        this.distance = parseFloat(content[FaceDetectionResult.INDEX_DISTANCE]);
        this.center = getVectorFromString(content[FaceDetectionResult.INDEX_CENTER]);
        this.rotationVector = getVectorFromString(content[FaceDetectionResult.INDEX_ROTATION]);
        this.position = getVectorFromString(content[FaceDetectionResult.INDEX_POSITION]);
        this.blinkScore = parseFloat(content[FaceDetectionResult.INDEX_BLINK_SCORE])

        this.landmarks = [];
        for (let i = FaceDetectionResult.INDEX_LANDMARKS; i < content.length; i++) {
            this.landmarks.push(FaceDetectionResult.getLandmarkFromString(content[i]));
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
}

export class FaceDetectionResults extends DetectionResults<FaceDetectionResult> {

    parse(content: string[]): FaceDetectionResult {
        return new FaceDetectionResult(content);
    }

}