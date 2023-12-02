import * as THREE from "three";
import { Face } from "./models/face";

export class GazeVector extends THREE.ArrowHelper {

  constructor(face: Face) {
    super(face.getGazeVector(), face.getCenter(), 5, 0xffffff);
  }

}
