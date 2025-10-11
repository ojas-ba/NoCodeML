import React, { createContext, useContext, useState, useEffect } from 'react';
import { authAPI } from '@/services/apiService';
import { useToast } from '@/components/ui/use-toast';

interface User {
  id: string;
  email: string;
}

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const { toast } = useToast();

  // Check for existing token on mount
  useEffect(() => {
    const initAuth = async () => {
      const storedToken = localStorage.getItem('auth_token');
      if (storedToken) {
        try {
          setToken(storedToken);
          const userData = await authAPI.getCurrentUser();
          setUser(userData);
        } catch (error) {
          localStorage.removeItem('auth_token');
          setToken(null);
        }
      }
      setIsLoading(false);
    };

    initAuth();
  }, []);

  const login = async (email: string, password: string) => {
    try {
      const response = await authAPI.login(email, password);
      const accessToken = response.access_token;
      
      if (!accessToken) {
        throw new Error('No access token received from server');
      }
      
      localStorage.setItem('auth_token', accessToken);
      setToken(accessToken);
      
      const userData = await authAPI.getCurrentUser();
      setUser(userData);
      
      toast({
        title: "Login successful",
        description: "Welcome back!",
      });
    } catch (error: any) {
      console.error('Login error:', error);
      const errorMessage = error?.response?.data?.detail || error?.message || "Invalid credentials";
      toast({
        title: "Login failed",
        description: errorMessage,
        variant: "destructive",
      });
      throw error;
    }
  };

  const register = async (email: string, password: string) => {
    try {
      await authAPI.register(email, password);
      
      toast({
        title: "Registration successful",
        description: "You can now log in with your credentials.",
      });
    } catch (error: any) {
      console.error('Registration error:', error);
      const errorMessage = error?.response?.data?.detail || error?.message || "Could not create account";
      toast({
        title: "Registration failed",
        description: errorMessage,
        variant: "destructive",
      });
      throw error;
    }
  };

  const logout = () => {
    localStorage.removeItem('auth_token');
    setToken(null);
    setUser(null);
    toast({
      title: "Logged out",
      description: "You have been logged out successfully.",
    });
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user,
        isLoading,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
