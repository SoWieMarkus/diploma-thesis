import * as THREE from "three";
import {Face} from "./models/face";
import {FacePointCloud} from "./point-cloud";
import {GazeVector} from "./gaze-vector";
import {HeadRotation} from "./head-rotation";
import Delaunator from "delaunator";

export class FaceMesh extends THREE.Mesh {

  constructor(private face: Face) {
    super();

    const surface = new THREE.Mesh(new SurfaceFaceGeoemtry(face), new THREE.MeshStandardMaterial({
      color: 0xaaaaaa,
      side: THREE.DoubleSide,
    }))
    const pointCloud = new FacePointCloud(face);
    const gazeVector = new GazeVector(face);
    const rotation = new HeadRotation(face);

    this.add(pointCloud);
    this.add(gazeVector);
    this.add(rotation);
    this.add(surface);
  }

  public getFace() {
    return this.face;
  }

}


export class SurfaceFaceGeoemtry extends THREE.BufferGeometry {

  constructor(face: Face) {
    super();
    const vectors: THREE.Vector3[] = [];

    for (const landmark of face.getLandmarks()) {
      vectors.push(landmark.getModel3DPosition());
    }

    this.setFromPoints(vectors);

    const delaunayIndex = Delaunator.from(vectors.map(vector => [vector.x, vector.y]));
    const bufferGeometryIndex = [];
    for (const element of delaunayIndex.triangles) {
      bufferGeometryIndex.push(element);
    }

    this.setIndex(bufferGeometryIndex);
    this.computeVertexNormals();
  }
}

