import { IVector, getVectorFromString } from "./helper";
import { DetectionResult, DetectionResults } from "./detection-results";

export class GazeDetectionResult extends DetectionResult {

    private static readonly INDEX_PITCH = 2;
    private static readonly INDEX_YAW = 3;
    private static readonly INDEX_VECTOR = 4;
    private static readonly INDEX_INTERSECTION = 5;
    private static readonly INDEX_HIT = 6;

    public pitch?: number;
    public yaw?: number;
    public gazeVector?: IVector;
    public intersection?: IVector;
    public hit?: boolean;

    constructor(content: string[]) {
        super(content);

        if (content.length === 2) return;

        this.pitch = parseFloat(content[GazeDetectionResult.INDEX_PITCH]);
        this.yaw = parseFloat(content[GazeDetectionResult.INDEX_YAW]);
        this.gazeVector = getVectorFromString(content[GazeDetectionResult.INDEX_VECTOR]);
        this.intersection = getVectorFromString(content[GazeDetectionResult.INDEX_INTERSECTION]);
        this.hit = content[GazeDetectionResult.INDEX_HIT] === "true" ? true : false;
    }

}

export class GazeDetectionResults extends DetectionResults<GazeDetectionResult> {

    parse(content: string[]): GazeDetectionResult {
        return new GazeDetectionResult(content);
    }

}