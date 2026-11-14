import { NavLink, Outlet, useParams } from "react-router";

export function Layout() {
  const { videoId } = useParams();
  return (
    <div className="app">
      <header className="topbar">
        <NavLink to="/" className="brand">
          <img src="/favicon.svg" alt="" />
          Gaze Estimator
        </NavLink>
        <nav>
          <NavLink to="/" end>
            Videos
          </NavLink>
          {videoId && (
            <>
              <NavLink to={`/videos/${videoId}/view`}>Viewer</NavLink>
              <NavLink to={`/videos/${videoId}/label`}>Labeling</NavLink>
            </>
          )}
        </nav>
        <div className="spacer" />
        <span className="muted">PrioBike · diploma thesis</span>
      </header>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
