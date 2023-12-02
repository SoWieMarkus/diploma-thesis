import * as THREE from "three";
import { FaceMesh } from "./scene/content/face-mesh";
import { FaceDataFrame } from "./scene/content/models/face";
import { PhoneMesh } from "./scene/content/phone";
import { Scene } from "./scene/scene.component";

export class LabelScene extends Scene {

  private static readonly PLANE_GEOMETRY = new THREE.PlaneGeometry(1, 1);
  private static readonly PLANE_MATERIAL = new THREE.MeshBasicMaterial({ color: 0x000000, side: THREE.DoubleSide, transparent: true, opacity: 0.3 });

  private faceMesh?: FaceMesh;
  private readonly phone: PhoneMesh;
  private readonly plane: THREE.Mesh;
  private readonly center: THREE.Mesh;
  private readonly intersection: THREE.Mesh;

  constructor() {
    super();

    this.phone = new PhoneMesh();
    this.plane = new THREE.Mesh(LabelScene.PLANE_GEOMETRY, LabelScene.PLANE_MATERIAL);

    const geometry = new THREE.BoxGeometry(.01, .01, .01);
    const material = new THREE.MeshBasicMaterial({ color: 0xB0C4DE, side: THREE.DoubleSide });
    this.center = new THREE.Mesh(geometry, material);

    const geometryIntersection = new THREE.BoxGeometry(.01, .01, .01);
    const materialIntersection = new THREE.MeshBasicMaterial({ color: 0xffff00, side: THREE.DoubleSide })
    this.intersection = new THREE.Mesh(geometryIntersection, materialIntersection)
  }

  public override initialize(): void {
    super.initialize();
    this.add(this.phone);
    this.add(this.plane);
    this.add(this.center);
    this.add(this.intersection);
    this.scale.y = -1;
    this.scale.x = -1;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5); // Add ambient light
    this.add(ambientLight);
  }

  public load(data?: FaceDataFrame) {
    if (this.faceMesh !== undefined)
      this.remove(this.faceMesh);

    this.phone.highlight(false);
    this.intersection.visible = false;
    if (data === undefined) return;

    const face = data.getFace();

    if (face === undefined) return;

    this.intersection.visible = true;
    this.intersection.position.x = face.getGazeIntersection().x;
    this.intersection.position.y = face.getGazeIntersection().y;
    this.intersection.position.z = face.getGazeIntersection().z;
    this.faceMesh = new FaceMesh(face);

    this.add(this.faceMesh);
    this.phone.highlight(face.looksOnScreen())

  }

  public getPhone() {
    return this.phone;
  }
}
