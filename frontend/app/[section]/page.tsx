import { notFound } from 'next/navigation';
import Studio from '../../components/Studio';
export function generateStaticParams() { return ['dashboard', 'generate', 'training', 'dataset', 'history', 'settings'].map(section => ({ section })); }
export default async function Page({ params }: { params: Promise<{ section: string }> }) {
  const { section } = await params;
  if (!['dashboard', 'generate', 'training', 'dataset', 'history', 'settings'].includes(section)) notFound();
  return <Studio page={section} />;
}
