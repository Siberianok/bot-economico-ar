interface Props {
  title: string;
  detail?: string;
}

export function DataState({ title, detail }: Props) {
  return (
    <div className="rounded-lg border border-dashed border-obs-border bg-obs-card2 p-5 text-sm">
      <div className="font-medium text-obs-text">{title}</div>
      {detail ? <div className="mt-1 text-obs-muted">{detail}</div> : null}
    </div>
  );
}

