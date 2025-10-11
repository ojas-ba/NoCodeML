import { Link } from "react-router-dom";
import { Upload, BarChart3, Zap, Download, Brain, TrendingUp, Sparkles, MessageSquare } from "lucide-react";
import { Button } from "@/components/ui/button";

const Home = () => {
  const features = [
    {
      icon: Upload,
      title: "Data Upload & Management",
      description: "Drag and drop CSV files up to 100MB. Automatic validation and preprocessing suggestions.",
      gradient: "from-primary to-primary-blue"
    },
    {
      icon: BarChart3,
      title: "Interactive Visualizations",
      description: "Build custom plots with Plotly. Correlation matrices, distributions, and more.",
      gradient: "from-primary-purple to-accent"
    },
    {
      icon: Brain,
      title: "Automated Model Training",
      description: "Train multiple models simultaneously. Compare performance metrics in real-time.",
      gradient: "from-primary-blue to-info"
    },
    {
      icon: Sparkles,
      title: "Smart Hyperparameter Tuning",
      description: "Automatic hyperparameter optimization for best model performance without manual configuration.",
      gradient: "from-accent to-primary-blue"
    },
    {
      icon: MessageSquare,
      title: "AI Assistant Chatbot",
      description: "Get instant help and insights with our integrated AI assistant for data analysis and ML guidance.",
      gradient: "from-primary-purple to-primary"
    },
    {
      icon: TrendingUp,
      title: "Real-time Results",
      description: "Live training monitoring with detailed evaluation metrics and model comparison.",
      gradient: "from-primary to-primary-purple"
    },
    {
      icon: Zap,
      title: "Fast & Intuitive",
      description: "Streamlined workflow from data upload to predictions in minutes, not hours.",
      gradient: "from-accent to-primary-purple"
    }
  ];
  
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 md:py-32">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-primary-purple/10 to-primary-blue/10 animate-pulse-glow"></div>
        <div className="container relative mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center space-y-8 animate-fade-in">
            <h1 className="text-5xl md:text-7xl font-bold gradient-text leading-tight">
              Machine Learning Without Code
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto">
              Upload your data, visualize patterns, train models with automated hyperparameter tuning, and get AI-powered insights - all through an intuitive interface
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-4">
              <Link to="/experiments">
                <Button size="lg" className="gradient-primary text-background font-semibold hover:opacity-90 transition-opacity px-8 py-6 text-lg">
                  Get Started
                </Button>
              </Link>
              <Link to="/experiments">
                <Button size="lg" variant="outline" className="border-primary text-primary hover:bg-primary/10 px-8 py-6 text-lg">
                  View Demo
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>
      
      {/* Features Grid */}
      <section className="py-20 bg-card/30">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16 animate-slide-up">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Everything You Need for <span className="gradient-text">ML Experiments</span>
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              From data exploration to automated model optimization with AI assistance, all in one platform
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto">
            {features.map((feature, index) => (
              <div
                key={index}
                className="card-hover bg-card border border-border rounded-xl p-6 space-y-4"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.gradient} flex items-center justify-center`}>
                  <feature.icon className="w-6 h-6 text-background" />
                </div>
                <h3 className="text-xl font-semibold">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
      
      {/* CTA Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto bg-gradient-to-br from-primary/20 via-primary-purple/20 to-primary-blue/20 rounded-2xl border border-primary/30 p-12 text-center space-y-6">
            <h2 className="text-3xl md:text-4xl font-bold">
              Ready to Build Your First Model?
            </h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Start experimenting with machine learning in minutes. Powered by automated hyperparameter tuning and AI assistance. No coding required.
            </p>
            <Link to="/experiments">
              <Button size="lg" className="gradient-primary text-background font-semibold hover:opacity-90 transition-opacity px-8 py-6 text-lg">
                Launch Playground
              </Button>
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;
