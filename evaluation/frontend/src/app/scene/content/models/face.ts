import * as THREE from "three";

export interface IVector {
  x: number;
  y: number;
  z: number;
}

export interface ILandmark {
  index: number;
  model3dPosition: IVector;
  normalizedPosition: IVector;
  imageX: number;
  imageY: number;
}



export interface IFaceDataFrame {
  position?: IVector;
  rotationVector?: IVector;
  center?: IVector;
  distance?: number;
  frameNumber: number;
  timestamp: number;
  landmarks?: ILandmark[];
}

export interface IGazeDataFrame {
  gazeVector?: IVector;
  pitch?: number;
  yaw?: number;
  intersection?: IVector;
  hit?: boolean;
}

export interface IFrameDetails {
  frameNumber: number;
  face?: IFaceDataFrame;
  gaze?: IGazeDataFrame;
  label: number;
}

export class Landmark {

  private model3dPosition: THREE.Vector3;
  private index: number;
  private imageX: number;
  private imageY: number;
  private normalizedPostion: THREE.Vector3;


  constructor(landmark: ILandmark) {
    this.model3dPosition = Face.getThreeVectorFromIVector(landmark.model3dPosition);
    this.normalizedPostion = Face.getThreeVectorFromIVector(landmark.normalizedPosition);
    this.imageX = landmark.imageX;
    this.imageY = landmark.imageY;
    this.index = landmark.index;
  }

  public getIndex() {
    return this.index;
  }

  public getModel3DPosition() {
    return this.model3dPosition;
  }
}

export class FaceDataFrame {

  private face?: Face;
  private frameNumber: number;
  private timestamp: number;

  constructor(data: IFaceDataFrame, gaze: IGazeDataFrame) {
    this.frameNumber = data.frameNumber;
    this.timestamp = data.timestamp;

    if (data.center === undefined) return;

    this.face = new Face(data, gaze);

  }

  public getFace() {
    return this.face;
  }

}


export class Face {

  private gazeVector: THREE.Vector3;
  private headRotation: THREE.Vector3;
  private headPosition: THREE.Vector3;
  private gazeIntersection: THREE.Vector3;
  private center: THREE.Vector3;
  private distance: number;
  private landmarks: Landmark[];
  private gazeHit: boolean;

  constructor(data: IFaceDataFrame, gaze: IGazeDataFrame) {

    this.gazeHit = gaze.hit!;

    this.gazeVector = Face.getThreeVectorFromIVector(gaze.gazeVector!);
    this.gazeIntersection = Face.getThreeVectorFromIVector(gaze.intersection!)
    this.headRotation = Face.getThreeVectorFromIVector(data.rotationVector!);
    this.center = Face.getThreeVectorFromIVector(data.center!);
    this.distance = data.distance!;
    this.headPosition = Face.getThreeVectorFromIVector(data.position!);

    this.landmarks = [];
    for (const landmark of data.landmarks!) {
      this.landmarks.push(new Landmark(landmark));
    }
  }

  public looksOnScreen() {
    return this.gazeHit;
  }

  public getNosePosition() {
    const noseIndex = this.landmarks.length === 68 ? 30 : 1;


    for (const landmark of this.landmarks) {
      if (landmark.getIndex() === noseIndex) {
        return landmark.getModel3DPosition();
      }
    }
    return undefined;
  }

  public getGazeIntersection() {
    return this.gazeIntersection;
  }

  public getChinPosition() {
    const chinIndex = this.landmarks.length === 68 ? 8 : 199;


    for (const landmark of this.landmarks) {
      if (landmark.getIndex() === chinIndex) {
        return landmark.getModel3DPosition();
      }
    }
    return undefined;
  }

  public static getThreeVectorFromIVector(vector: IVector) {
    return new THREE.Vector3(vector.x, vector.y, vector.z);
  }

  public getHeadRotation() {
    return this.headRotation;
  }

  public getLandmarks() {
    return this.landmarks;
  }

  public getGazeVector() {
    return this.gazeVector;
  }

  public getCenter() {
    return this.center;
  }

}
