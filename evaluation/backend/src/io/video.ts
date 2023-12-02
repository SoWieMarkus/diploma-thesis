import path from "path";

import { Directory, getSubDirectories, getFilesWithExtension } from "./helper";
import { FaceDetectionResults, FaceDetectionResult } from "./face-frame";
import { GazeDetectionResults, GazeDetectionResult } from "./gaze-frame";

export interface IFrameDetails {
    frameNumber: number;
    face: FaceDetectionResult | undefined;
    gaze: GazeDetectionResult | undefined;
    label: undefined | number;
}


export class Video {

    private faceDetectionResults = new Map<string, FaceDetectionResults>();
    private gazeDetectionResults = new Map<string, GazeDetectionResults>();

    constructor(private directory: Directory) {
        this.faceDetectionResults = this.readFaceData();
        this.gazeDetectionResults = this.readGazeData();

        // TODO merge
    }

    private readFaceData() {
        const faceDirectoryPath = path.join(this.directory.path, "face")
        const faceDirectories = getSubDirectories(faceDirectoryPath);

        const faceDetectionResults = new Map<string, FaceDetectionResults>()

        for (const directory of faceDirectories) {
            const faceFilePath = path.join(faceDirectoryPath, `${directory.name}/${directory.name}.face`)
            const faceDetectionResult = new FaceDetectionResults(directory.name, faceFilePath);
            faceDetectionResults.set(faceDetectionResult.getModelName(), faceDetectionResult)
        }

        return faceDetectionResults;
    }

    private readGazeData() {
        const gazeDirectoryPath = path.join(this.directory.path, "gaze")
        const gazeDetectionResults = new Map<string, GazeDetectionResults>()
        const gazeFiles = getFilesWithExtension(gazeDirectoryPath, "gaze")

        for (const directory of gazeFiles) {
            const gazeDetectionResult = new GazeDetectionResults(directory.name.replace(".gaze", ""), directory.path);
            gazeDetectionResults.set(gazeDetectionResult.getModelName(), gazeDetectionResult)
        }

        return gazeDetectionResults;
    }

    public getByFrameNumber(faceModel: string, gazeModel: string, frameNumber: number): IFrameDetails | undefined {
        const faceModelResult = this.faceDetectionResults.get(faceModel);
        const gazeModelResult = this.gazeDetectionResults.get(`${faceModel}_${gazeModel}`);

        if (faceModelResult === undefined || gazeModelResult === undefined) return undefined;

        const faceResult = faceModelResult.getByFrameNumber(frameNumber);
        const gazeResult = gazeModelResult.getByFrameNumber(frameNumber);

        return {
            "frameNumber": frameNumber,
            "face": faceResult,
            "gaze": gazeResult,
            "label": undefined
        }

    }

    public getName() {
        return this.directory.name;
    }

    public getSize(face: string, gaze: string) {
        const gazeModel = this.gazeDetectionResults.get(`${face}_${gaze}`)
        return gazeModel?.getSize();
    }

    public getDetailsGaze() {

        const listOfVideos = [];

        for (const result of this.gazeDetectionResults.values()) {
            const parts = result.getModelName().split("_");
            const face = parts[0];
            const gaze = parts[1];
            const size = result.getSize();
            listOfVideos.push({
                face: face,
                gaze: gaze,
                frames: size
            });
        }

        return listOfVideos;

    }

}