interface TabItem {
  key: string;
  label: string;
}

interface Props {
  items: TabItem[];
  value: string;
  onChange: (value: string) => void;
  label: string;
}

export function SegmentedTabs({ items, value, onChange, label }: Props) {
  return (
    <div
      className="flex max-w-full gap-1 overflow-x-auto rounded-md border border-obs-border bg-obs-card p-1 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
      role="tablist"
      aria-label={label}
    >
      {items.map((item) => (
        <button
          key={item.key}
          type="button"
          role="tab"
          aria-selected={value === item.key}
          className={`min-h-9 shrink-0 rounded px-3 text-sm transition ${
            value === item.key
              ? "bg-obs-card2 text-obs-blue"
              : "text-obs-muted hover:text-obs-text"
          }`}
          onClick={() => onChange(item.key)}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
