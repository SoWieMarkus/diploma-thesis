export interface ILabelDetails {
    id: string;
    progress: number;
    total: number;
}

export abstract class Label {

    public abstract getDetails(): ILabelDetails;

}