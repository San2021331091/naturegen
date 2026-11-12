'use client';
import { Mountain } from 'lucide-react';

export function Empty({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="empty">
      <span className="empty-icon"><Mountain size={28} strokeWidth={1.3} /></span>
      <h3>{title}</h3>
      <div className="empty-desc">{children}</div>
    </div>
  );
}
