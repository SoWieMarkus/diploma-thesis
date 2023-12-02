import * as THREE from "three";
import {Face} from "./models/face";

export class HeadRotation extends THREE.AxesHelper {

  constructor(face: Face) {
    super(.2);

    const origin = face.getNosePosition()!;
    const rotation = face.getHeadRotation();

    const rotationEuler = new THREE.Euler().setFromVector3(rotation);

    this.setRotationFromEuler(rotationEuler);

    this.position.x = origin.x;
    this.position.y = origin.y;
    this.position.z = origin.z;

    this.scale.x = -1;

    this.scale.z = -3;

  }

}
