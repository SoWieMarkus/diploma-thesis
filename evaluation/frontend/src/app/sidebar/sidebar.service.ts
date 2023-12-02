import { Injectable } from "@angular/core";
import { BehaviorSubject } from "rxjs";

@Injectable({
  providedIn: "root"
})
export class SidebarService {
  private currentOpenState = false;

  private isOpened: BehaviorSubject<boolean> = new BehaviorSubject(this.currentOpenState);

  private observableOpened = this.isOpened.asObservable();

  public get isOpen() {
    return this.observableOpened;
  }

  toggleOpen = () => {
    this.currentOpenState = !this.currentOpenState;
    this.isOpened.next(this.currentOpenState);
  };
}
