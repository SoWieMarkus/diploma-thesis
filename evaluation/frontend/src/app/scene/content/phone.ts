import * as THREE from "three";


export class PhoneMesh extends THREE.Mesh {

  public static HEIGHT = .2;
  public static POSITION = 0;

  constructor() {
    super(new THREE.BoxGeometry(PhoneMesh.HEIGHT, PhoneMesh.HEIGHT, 0.01), new THREE.MeshBasicMaterial({ color: 0xf75e25, side: THREE.DoubleSide }));
    this.position.y = PhoneMesh.HEIGHT / 2;
  }

  public highlight(hit: boolean) {
    const material = this.material as THREE.MeshBasicMaterial;
    material.color = hit ? new THREE.Color(0x00ff00) : new THREE.Color(0x000000);
  }

  public update() {
    this.geometry = new THREE.BoxGeometry(PhoneMesh.HEIGHT, PhoneMesh.HEIGHT, 0.01)
    this.position.y = PhoneMesh.HEIGHT / 2 - PhoneMesh.POSITION;
  }

}
