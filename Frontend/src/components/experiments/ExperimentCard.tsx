import { Beaker, Calendar, Trash2, Copy, FolderOpen, MoreVertical } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useNavigate } from "react-router-dom";

interface ExperimentCardProps {
  experiment: {
    id: string;
    name: string;
    datasetName?: string;
    status: "in_progress" | "completed";
    updatedAt: string;
    config?: {
      taskType?: string;
    };
  };
  onDelete: (id: string) => void;
  onDuplicate: (id: string) => void;
}

const ExperimentCard = ({ experiment, onDelete, onDuplicate }: ExperimentCardProps) => {
  const navigate = useNavigate();

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric', 
      year: 'numeric' 
    });
  };

  return (
    <div className="card-hover bg-card border border-border rounded-xl p-6">
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-4 flex-1">
          <div className="w-12 h-12 rounded-lg gradient-primary flex items-center justify-center flex-shrink-0">
            <Beaker className="w-6 h-6 text-background" />
          </div>
          
          <div className="space-y-2 flex-1 min-w-0">
            <div className="flex items-center justify-between gap-4">
              <h3 className="text-xl font-semibold truncate">{experiment.name}</h3>
              <span className={`px-2 py-0.5 rounded text-xs font-medium whitespace-nowrap ${
                experiment.status === "completed" 
                  ? "bg-success/20 text-success" 
                  : "bg-warning/20 text-warning"
              }`}>
                {experiment.status === "completed" ? "Completed" : "In Progress"}
              </span>
            </div>
            
            <div className="flex items-center space-x-4 text-sm text-muted-foreground">
              <span className="flex items-center">
                <Calendar className="w-4 h-4 mr-1" />
                {formatDate(experiment.updatedAt)}
              </span>
              {experiment.datasetName && (
                <span className="truncate">{experiment.datasetName}</span>
              )}
              {experiment.config?.taskType && (
                <span className="px-2 py-0.5 rounded bg-primary/20 text-primary text-xs font-medium">
                  {experiment.config.taskType}
                </span>
              )}
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-2 ml-4">
          <Button 
            variant="outline"
            onClick={() => navigate(`/playground/${experiment.id}`)}
            className="gap-2"
          >
            <FolderOpen className="w-4 h-4" />
            Open
          </Button>
          
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon">
                <MoreVertical className="w-4 h-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => onDuplicate(experiment.id)}>
                <Copy className="w-4 h-4 mr-2" />
                Duplicate
              </DropdownMenuItem>
              <DropdownMenuItem 
                onClick={() => onDelete(experiment.id)}
                className="text-destructive"
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Delete
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </div>
  );
};

export default ExperimentCard;
