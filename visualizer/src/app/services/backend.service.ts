import { HttpClient } from "@angular/common/http";
import { Injectable, inject } from "@angular/core";
import type { Observable } from "rxjs";
import type * as z from "zod";

const BACKEND_URL = "http://localhost:3000/api";

@Injectable({
  providedIn: "root",
})
export class BackendService {
  private readonly http = inject(HttpClient);

  private handleResponse<T>(observable: Observable<T>, responseModel?: z.Schema<T>): Promise<T> {
    return new Promise((resolve, reject) => {
      observable.subscribe({
        next: (response) => {
          if (!responseModel) {
            resolve(response);
            return;
          }
          const { success, data, error } = responseModel.safeParse(response);
          if (!success) {
            reject(new Error(error.message));
            return;
          }
          resolve(data);
        },
        error: (error) => {
          console.error("Error in backend service:", error);
          reject(new Error(error.error?.error ?? "An error occurred while processing the request."));
        },
      });
    });
  }

  private get<T>(url: string, responseModel?: z.Schema<T>): Promise<T> {
    return this.handleResponse(this.http.get<T>(`${BACKEND_URL}/${url}`), responseModel);
  }

  private post<T>(url: string, body: unknown, responseModel?: z.Schema<T>): Promise<T> {
    return this.handleResponse(this.http.post<T>(`${BACKEND_URL}/${url}`, body), responseModel);
  }

  private delete<T>(url: string, responseModel?: z.Schema<T>): Promise<T> {
    return this.handleResponse(this.http.delete<T>(`${BACKEND_URL}/${url}`), responseModel);
  }

  public get video() {
    return {
      // TODO add model
      list: () => this.get("videos"),
      get: (id: string) => this.get(`videos/${id}`),
    }
  }
}
