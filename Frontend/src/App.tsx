import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import { ExperimentProvider } from "./contexts/ExperimentContext";
import { ModelsProvider } from "./contexts/ModelsContext";
import { TrainingProvider } from "./contexts/TrainingContext";
import ProtectedRoute from "./components/ProtectedRoute";
import Header from "./components/Header";
import Home from "./pages/Home";
import Datasets from "./pages/Datasets";
import Experiments from "./pages/Experiments";
import Playground from "./pages/Playground";
import Login from "./pages/Login";
import Register from "./pages/Register";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <AuthProvider>
        <ModelsProvider>
          <ExperimentProvider>
            <TrainingProvider>
            <Toaster />
            <Sonner />
          <BrowserRouter>
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />
              <Route
                path="/*"
                element={
                  <ProtectedRoute>
                    <div className="min-h-screen bg-background">
                      <Header />
                      <Routes>
                        <Route path="/" element={<Home />} />
                        <Route path="/datasets" element={<Datasets />} />
                        <Route path="/experiments" element={<Experiments />} />
                        <Route path="/playground/:experimentId" element={<Playground />} />
                        <Route path="*" element={<NotFound />} />
                      </Routes>
                    </div>
                  </ProtectedRoute>
                }
              />
            </Routes>
          </BrowserRouter>
            </TrainingProvider>
          </ExperimentProvider>
        </ModelsProvider>
      </AuthProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
