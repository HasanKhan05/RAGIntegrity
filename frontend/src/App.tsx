import { Navigate, Route, Routes } from "react-router-dom";
import Header from "./components/Header";
import About from "./pages/About";
import Demo from "./pages/Demo";
import Documents from "./pages/Documents";
import Results from "./pages/Results";
import "./styles.css";

export default function App() {
  return (
    <div className="app-shell">
      <Header />
      <Routes>
        <Route path="/" element={<About />} />
        <Route path="/demo" element={<Demo />} />
        <Route path="/documents" element={<Documents />} />
        <Route path="/results" element={<Results />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </div>
  );
}
