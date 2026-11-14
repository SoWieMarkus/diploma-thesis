import {
  GizmoHelper,
  GizmoViewport,
  Grid,
  Line,
  OrbitControls,
  RoundedBox,
} from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";
import * as THREE from "three";
import type { Camera, Face, Vec3 } from "../api/schemas";
import { modelColor, COLORS } from "../lib/colors";
import { type Display, intersectCameraPlane, onDisplay, toThree } from "../lib/geometry";

/** Rotation matrix (OpenCV coordinates) from a Rodrigues vector. */
function rotationFromVector(r: Vec3): THREE.Matrix4 {
  const angle = Math.hypot(...r);
  if (angle === 0) return new THREE.Matrix4();
  return new THREE.Matrix4().makeRotationAxis(
    new THREE.Vector3(r[0] / angle, r[1] / angle, r[2] / angle),
    angle,
  );
}

/** Smartphone on the handlebar with the virtual display, highlighted when a gaze hits it. */
function Phone({ display, active }: { display: Display; active: boolean }) {
  const { width, height, offset } = display;
  // Center of the display in OpenCV y is (height - 2 * offset) / 2, flipped for three.js.
  const centerY = -(height / 2 - offset);
  const color = active ? COLORS.positive : "#334155";
  return (
    <group>
      <RoundedBox
        args={[width + 0.01, height + 0.014, 0.008]}
        radius={0.004}
        smoothness={4}
        position={[0, centerY, 0.0045]}
      >
        <meshStandardMaterial color="#0f172a" metalness={0.6} roughness={0.35} />
      </RoundedBox>
      <mesh position={[0, centerY, -0.0002]} rotation={[0, Math.PI, 0]}>
        <planeGeometry args={[width, height]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={active ? 0.9 : 0.25}
          transparent
          opacity={0.85}
          side={THREE.DoubleSide}
        />
      </mesh>
      {/* front camera */}
      <mesh position={[0, 0, -0.001]}>
        <sphereGeometry args={[0.0025, 24, 24]} />
        <meshStandardMaterial color="#5eead4" emissive="#5eead4" emissiveIntensity={1.5} />
      </mesh>
    </group>
  );
}

function CameraFrustum({ camera, depth = 0.25 }: { camera: Camera; depth?: number }) {
  const corners = useMemo(() => {
    const corner = (u: number, v: number) =>
      toThree([
        ((u - camera.cx) / camera.fx) * depth,
        ((v - camera.cy) / camera.fy) * depth,
        depth,
      ]);
    return [
      corner(0, 0),
      corner(camera.width, 0),
      corner(camera.width, camera.height),
      corner(0, camera.height),
    ];
  }, [camera, depth]);
  const origin: [number, number, number] = [0, 0, 0];
  return (
    <group>
      {corners.map((c, i) => (
        <Line
          key={i}
          points={[origin, c]}
          color="#5eead4"
          lineWidth={1}
          transparent
          opacity={0.35}
        />
      ))}
      <Line
        points={[...corners, corners[0]]}
        color="#5eead4"
        lineWidth={1}
        transparent
        opacity={0.35}
      />
    </group>
  );
}

function FaceModel({ face }: { face: Face }) {
  const positions = useMemo(
    () => new Float32Array((face.landmarks_3d ?? []).flatMap((p) => toThree(p))),
    [face.landmarks_3d],
  );
  const axes = useMemo(() => {
    if (!face.rotation_vector || !face.center) return [];
    const m = rotationFromVector(face.rotation_vector).elements; // column-major
    const c = face.center;
    const column = (i: number): Vec3 => [m[i * 4], m[i * 4 + 1], m[i * 4 + 2]];
    return [0, 1, 2].map((i) => {
      const axis = column(i);
      const end: Vec3 = [c[0] + axis[0] * 0.05, c[1] + axis[1] * 0.05, c[2] + axis[2] * 0.05];
      return { end: toThree(end), color: ["#ef4444", "#22c55e", "#3b82f6"][i] };
    });
  }, [face.rotation_vector, face.center]);

  if (!face.center) return null;
  const center = toThree(face.center);
  return (
    <group>
      <points key={face.id}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        </bufferGeometry>
        <pointsMaterial size={0.0035} color="#99f6e4" sizeAttenuation transparent opacity={0.9} />
      </points>
      <mesh position={center}>
        <sphereGeometry args={[0.004, 16, 16]} />
        <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.6} />
      </mesh>
      {axes.map((a, i) => (
        <Line key={i} points={[center, a.end]} color={a.color} lineWidth={2} />
      ))}
    </group>
  );
}

function GazeRay({
  center,
  vector,
  color,
  highlighted,
  display,
}: {
  center: Vec3;
  vector: Vec3;
  color: string;
  highlighted: boolean;
  display: Display;
}) {
  const hit = intersectCameraPlane(center, vector);
  const usable = hit && Math.hypot(hit[0], hit[1]) < 1.5;
  const end: Vec3 = usable
    ? hit
    : [center[0] + vector[0] * 0.6, center[1] + vector[1] * 0.6, center[2] + vector[2] * 0.6];
  const onPhone = onDisplay(hit, display);
  return (
    <group>
      <Line
        points={[toThree(center), toThree(end)]}
        color={color}
        lineWidth={highlighted ? 3 : 1.5}
        transparent
        opacity={highlighted ? 1 : 0.55}
        dashed={!highlighted}
        dashSize={0.01}
        gapSize={0.006}
      />
      {usable && (
        <mesh position={toThree(hit)}>
          <sphereGeometry args={[highlighted ? 0.005 : 0.0035, 20, 20]} />
          <meshStandardMaterial
            color={onPhone ? COLORS.positive : color}
            emissive={onPhone ? COLORS.positive : color}
            emissiveIntensity={1.2}
          />
        </mesh>
      )}
    </group>
  );
}

interface Props {
  camera: Camera;
  face?: Face;
  display: Display;
  gazeModels: string[];
  highlightModel?: string;
  showFrustum?: boolean;
}

export function GazeScene({
  camera,
  face,
  display,
  gazeModels,
  highlightModel,
  showFrustum = true,
}: Props) {
  const highlighted = face?.gazes.find((g) => g.model === highlightModel);
  const active = Boolean(
    face?.center &&
    highlighted &&
    onDisplay(intersectCameraPlane(face.center, highlighted.vector), display),
  );

  return (
    <Canvas camera={{ position: [0.42, 0.2, -0.58], fov: 45, near: 0.005, far: 50 }} dpr={[1, 2]}>
      <color attach="background" args={["#0b0f17"]} />
      <fog attach="fog" args={["#0b0f17", 2.5, 7]} />
      <ambientLight intensity={0.45} />
      <directionalLight position={[1, 2, 1.5]} intensity={1.2} />
      <pointLight position={[0, 0.2, -0.4]} intensity={0.6} color="#5eead4" />

      <Grid
        position={[0, -0.32, 0]}
        args={[10, 10]}
        cellSize={0.05}
        cellThickness={0.6}
        cellColor="#1e293b"
        sectionSize={0.25}
        sectionThickness={1.1}
        sectionColor="#0f766e"
        fadeDistance={5}
        fadeStrength={1.6}
        infiniteGrid
      />

      <Phone display={display} active={active} />
      {showFrustum && <CameraFrustum camera={camera} />}

      {face?.detected && <FaceModel face={face} />}
      {face?.center &&
        face.gazes
          .filter((g) => gazeModels.includes(g.model))
          .map((g) => (
            <GazeRay
              key={g.id}
              center={face.center!}
              vector={g.vector}
              color={modelColor(g.model)}
              highlighted={g.model === highlightModel}
              display={display}
            />
          ))}

      <OrbitControls makeDefault target={[0, 0, -0.18]} enableDamping dampingFactor={0.08} />
      <GizmoHelper alignment="bottom-right" margin={[70, 70]}>
        <GizmoViewport axisColors={["#ef4444", "#22c55e", "#3b82f6"]} labelColor="#0b0f17" />
      </GizmoHelper>
    </Canvas>
  );
}
