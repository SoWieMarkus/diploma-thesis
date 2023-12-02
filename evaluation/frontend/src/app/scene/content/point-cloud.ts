import * as THREE from "three";
import { Face } from "./models/face";

export class FacePointCloud extends THREE.Points {

  constructor(face: Face) {
    super(new FacePointCloudGeometry(face), new THREE.PointsMaterial({ color: 0x888888, size: 0.005 }));
  }

}

class FacePointCloudGeometry extends THREE.BufferGeometry {

  private indexAttribute: THREE.BufferAttribute;
  private positionAttribute: THREE.BufferAttribute;
  private colorAttribute: THREE.BufferAttribute;

  constructor(face: Face) {
    super();

    const colors = [];
    const positions = [];
    const indexLandmark = [];

    for (const landmark of face.getLandmarks()) {
      colors.push(0, 0, 0);
      const position = landmark.getModel3DPosition();
      positions.push(position.x, position.y, position.z);

      const index = landmark.getIndex();
      indexLandmark.push(index);
    }

    this.positionAttribute = new THREE.BufferAttribute(new Float32Array(positions), 3);
    this.colorAttribute = new THREE.BufferAttribute(new Float32Array(colors), 3);
    this.indexAttribute = new THREE.BufferAttribute(new Float32Array(indexLandmark), 1);


    this.setAttribute("position", this.positionAttribute);
    this.setAttribute("landmarkIndex", this.indexAttribute);
    this.setAttribute("color", this.colorAttribute);

  }

}

