import { Routes } from '@angular/router';
import { DashboardPage } from './pages/dashboard-page/dashboard-page';

export const routes: Routes = [
  {
    path: "",
    title: "Visualizer: Dashboard",
    component: DashboardPage,
  },
  {
    path: "**",
    redirectTo: "/",
    pathMatch: "full",
  },
];
