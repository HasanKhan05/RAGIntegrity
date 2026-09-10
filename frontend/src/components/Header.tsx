import { NavLink } from "react-router-dom";

const pages = [["About", "/"], ["Demo", "/demo"], ["Documents", "/documents"], ["Results", "/results"]] as const;

export default function Header() {
  return (
    <header className="site-header">
      <div className="brand"><strong>RAGIntegrity</strong><span>Evaluating Retrieval Poisoning Attacks and Defenses</span></div>
      <nav aria-label="Primary navigation">
        {pages.map(([label, path]) => <NavLink key={path} to={path} end={path === "/"}>{label}</NavLink>)}
      </nav>
    </header>
  );
}
