import { AfterViewInit, Component, ElementRef, Input, OnDestroy, ViewChild } from "@angular/core";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";

export interface CameraConfig {
  fov: number;
  far: number;
  aspect: number;
  near: number;
}

export class Scene extends THREE.Scene {
  protected renderer?: THREE.WebGLRenderer;
  protected raycaster: THREE.Raycaster;
  protected camera?: THREE.PerspectiveCamera;
  protected controls?: OrbitControls;

  protected mouseCoordinates: THREE.Vector2;
  private pointer: THREE.Vector2;

  private onIntersect: (intersection?: THREE.Intersection) => void;
  private objectsToIntersect: THREE.Object3D<THREE.Event>[];

  constructor() {
    super();

    this.raycaster = new THREE.Raycaster();

    this.mouseCoordinates = new THREE.Vector2();
    this.pointer = new THREE.Vector2();
    this.objectsToIntersect = [];
    this.onIntersect = () => {
      console.info("Scene has no implementation of intersections.");
    };
  }

  public applyCameraConfig(fov: number, near: number, far: number, aspect: number) {
    this.camera = new THREE.PerspectiveCamera(fov, aspect, near, far);
  }


  public getCamera() {
    if (this.camera === undefined) throw new Error("Illegal State");

    return this.camera;
  }

  public getControls() {
    if (this.controls === undefined) throw new Error("Illegal State");
    return this.controls;
  }

  public resize(width: number, height: number) {
    if (this.camera === undefined || this.renderer === undefined) throw new Error("Illegal State.");
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  public initializeRenderer(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      canvas: canvas
    });
  }

  public override remove(...objects: THREE.Object3D<THREE.Event>[]) {
    for (const object of objects) {
      this.removeObjectFromObjectsToIntersect(object);
      super.remove(object);
    }
    return this;
  }

  protected removeObjectFromObjectsToIntersect(object: THREE.Object3D<THREE.Event>) {
    if (this.objectsToIntersect.includes(object)) {
      const index = this.objectsToIntersect.indexOf(object, 0);
      if (index > -1) {
        this.objectsToIntersect.splice(index, 1);
      }
    }
  }

  public addObjectToIntersect(object: THREE.Object3D<THREE.Event>) {
    if (this.objectsToIntersect.includes(object)) return;
    this.objectsToIntersect.push(object);
  }

  public setObjectsToIntersect(objects: THREE.Object3D<THREE.Event>[]) {
    this.objectsToIntersect = objects;
  }

  public setOnIntersectionListener(observer: (intersection?: THREE.Intersection) => void) {
    this.onIntersect = observer;
  }

  public getObjectsBy(attribute: string, value: unknown) {
    const objects: THREE.Object3D<THREE.Event>[] = [];
    this.traverse(function (child) {
      const str = attribute as keyof typeof child;
      if (child[str] === value) {
        objects.push(child);
      }
    });
    return objects;
  }


  public initialize(): void {
    if (this.camera === undefined || this.renderer === undefined) throw new Error("Illegal State.");
    this.clear();
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
  }

  protected onRender() {
    return;
  }

  protected render() {
    if (this.camera === undefined || this.renderer === undefined) throw new Error("Illegal State.");
    this.renderer.render(this, this.camera);

    if (this.objectsToIntersect.length > 0) {
      this.lookForIntersections();
    }

    this.onRender();
  }

  private lookForIntersections() {
    if (this.camera === undefined) throw new Error("Illegal State");
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const intersections = this.raycaster.intersectObjects(this.objectsToIntersect, false);
    if (intersections.length === 0) {
      this.onIntersect(undefined);
      return;
    }

    this.onIntersect(this.chooseIntersection(intersections));
  }

  protected chooseIntersection(
    intersections: THREE.Intersection[]
  ): THREE.Intersection<THREE.Object3D<THREE.Event>> | undefined {
    return intersections[0];
  }

  public getRenderer() {
    if (this.renderer === undefined) throw new Error("Illegal State");
    return this.renderer;
  }

  public setMousePosition(x: number, y: number) {
    this.mouseCoordinates.x = x;
    this.mouseCoordinates.y = y;
  }

  public setPointer(x: number, y: number) {
    this.pointer.x = x;
    this.pointer.y = y;
  }

  public setCameraFocusX(x: number) {
    if (this.controls === undefined || this.camera === undefined) throw new Error("Illegal State");
    this.controls.target.x = x;
    this.controls.update();
    const cameraTarget = new THREE.Vector3();
    this.controls.target.copy(cameraTarget);
    this.camera.lookAt(cameraTarget);
  }

  public setCameraFocusY(y: number) {
    if (this.controls === undefined || this.camera === undefined) throw new Error("Illegal State");
    this.controls.target.y = y;
    this.controls.update();
    const cameraTarget = new THREE.Vector3();
    this.controls.target.copy(cameraTarget);
    this.camera.lookAt(cameraTarget);
  }

  public setCameraFocusZ(z: number) {
    if (this.controls === undefined || this.camera === undefined) throw new Error("Illegal State");

    this.controls.target.z = z;
    this.controls.update();
    const cameraTarget = new THREE.Vector3();
    this.controls.target.copy(cameraTarget);
    this.camera.lookAt(cameraTarget);
  }

  public setCameraFocus(x: number, y: number, z: number) {
    if (this.controls === undefined || this.camera === undefined) throw new Error("Illegal State");

    this.camera.lookAt(new THREE.Vector3(x, y, z));
    this.controls.target = new THREE.Vector3(x, y, z);
    this.controls.update();
  }

  public toggleVisible(name: string, visible: boolean) {
    const meshes = this.getObjectsByProperty("name", name);
    for (const mesh of meshes) {
      mesh.visible = visible;
    }
  }

  public startRendering() {
    if (this.renderer === undefined) throw new Error("illegal state")
    this.renderer.setAnimationLoop(() => this.render());
  }

  public getMouseCoordinates() {
    return this.mouseCoordinates;
  }
}

@Component({
  selector: "app-scene",
  templateUrl: "./scene.component.html",
  styleUrls: ["./scene.component.css"]
})
export class SceneComponent implements AfterViewInit, OnDestroy {
  @Input() public scene!: Scene;

  @Input() public cameraNear = 0.1;
  @Input() public cameraFOV = 75;
  @Input() public cameraFar = 5000;

  @Input() public backgroundColor = 0x303030;

  @ViewChild("canvas")
  private canvasRef!: ElementRef;

  onMouseMove(event: MouseEvent) {
    this.scene.setMousePosition(event.clientX, event.clientY);
    this.scene.setPointer(
      (event.offsetX / this.canvas.clientWidth) * 2 - 1,
      -(event.offsetY / this.canvas.clientHeight) * 2 + 1
    );
  }

  ngOnDestroy(): void {
    this.scene.getRenderer().dispose();
  }

  ngAfterViewInit(): void {
    const observer = new ResizeObserver(() => {
      this.scene.resize(this.canvas.clientWidth, this.canvas.clientHeight);
      this.canvas.style.width = "100%";
      this.canvas.style.height = "100%";
    });
    observer.observe(this.canvas);

    this.updateCamera();
    this.scene.initializeRenderer(this.canvas);
    this.scene.initialize();
    this.scene.startRendering();
    this.scene.background = new THREE.Color(this.backgroundColor);
  }

  private updateCamera() {
    this.scene.applyCameraConfig(this.cameraFOV, this.cameraNear, this.cameraFar, this.aspectRatio);
  }

  private get canvas(): HTMLCanvasElement {
    return this.canvasRef.nativeElement;
  }

  private get aspectRatio(): number {
    return this.canvas.clientWidth / this.canvas.clientHeight;
  }
}
