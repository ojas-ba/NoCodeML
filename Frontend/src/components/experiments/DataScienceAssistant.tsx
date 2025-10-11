import { useState, useEffect, useRef } from 'react';
import { X, MessageCircle, Send, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { useEDA } from '@/hooks/useEDA';
import { useExperiment } from '@/contexts/ExperimentContext';
import ReactMarkdown from 'react-markdown';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface DataScienceAssistantProps {
  datasetId?: string;
  edaData?: any; // EDA data can be passed directly or fetched via hook
  currentPhase?: 'analysis' | 'config' | 'training' | 'results' | 'predict';
  experimentConfig?: any; // Current experiment configuration
  trainingData?: any; // Training runs and metrics
  resultsData?: any; // Final results and model comparisons
}

export const DataScienceAssistant = ({ 
  datasetId, 
  edaData: propEdaData,
  currentPhase = 'analysis',
  experimentConfig,
  trainingData,
  resultsData 
}: DataScienceAssistantProps) => {
  
  // Determine welcome message based on phase
  const getWelcomeMessage = () => {
    switch (currentPhase) {
      case 'analysis':
        return '👋 Hello! I\'m your Data Science Assistant. I can help you understand your data quality, distributions, and what insights your dataset reveals. Ask me anything!';
      case 'config':
        return '🎯 Hi! I can help you choose the best models, select important features, and configure your experiment. What would you like to know?';
      case 'training':
        return '🚀 Hey! I\'m here to help you understand training progress, interpret metrics, and troubleshoot any issues. Ask away!';
      case 'results':
        return '📊 Hello! I can help you interpret model performance, compare results, and understand which model works best for your use case.';
      case 'predict':
        return '🔮 Hi! I can help you understand predictions, confidence scores, and how to use your trained models.';
      default:
        return '👋 Hello! I\'m your Data Science Assistant. Ask me anything about your ML experiment!';
    }
  };

  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: getWelcomeMessage()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [chatSize, setChatSize] = useState({ width: 384, height: 500 }); // Default: w-96 (384px), h-[500px]
  const [isResizing, setIsResizing] = useState(false);
  const [resizeDirection, setResizeDirection] = useState<'corner-br' | 'corner-bl' | 'corner-tr' | 'corner-tl' | 'bottom' | 'top' | 'right' | 'left' | null>(null);
  const [resizeStart, setResizeStart] = useState({ x: 0, y: 0, width: 0, height: 0 });
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLDivElement>(null);
  const chatRef = useRef<HTMLDivElement>(null);

  const { currentExperiment } = useExperiment();
  const { edaData: hookEdaData } = useEDA(datasetId || currentExperiment?.datasetId);

  // Use provided EDA data or fetched data
  const edaData = propEdaData || hookEdaData;
  
  // Gemini API Configuration
  const GEMINI_API_KEY = import.meta.env.VITE_GEMINI_API_KEY || '';
  const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:streamGenerateContent?alt=sse&key=${GEMINI_API_KEY}`;
  
  // Update welcome message when phase changes
  useEffect(() => {
    setMessages([{
      role: 'assistant',
      content: getWelcomeMessage()
    }]);
  }, [currentPhase]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Generate phase-specific context for the AI
  const generatePhaseContext = () => {
    let context = '';

    // Always include experiment basics if available
    if (currentExperiment) {
      context += `# Current Experiment: ${currentExperiment.name}\n`;
      context += `Dataset: ${currentExperiment.datasetName}\n\n`;
    }

    // Phase-specific context
    switch (currentPhase) {
      case 'analysis':
        context += generateEDAContext();
        break;
      case 'config':
        context += generateConfigContext();
        break;
      case 'training':
        context += generateTrainingContext();
        break;
      case 'results':
        context += generateResultsContext();
        break;
      case 'predict':
        context += generatePredictionContext();
        break;
    }

    return context;
  };

  // Generate comprehensive data analysis inferences from EDA data
  const generateEDAContext = () => {
    if (!edaData) {
      return 'No data analysis available yet. Please complete the EDA step first.';
    }

    const { dataset_info, columns, numeric_columns, categorical_columns, statistics, correlations, missing_data_summary } = edaData;

    let inferences = `# Dataset Analysis Summary

## Dataset Overview
- **Name**: ${dataset_info.name}
- **Total Rows**: ${dataset_info.row_count.toLocaleString()}
- **Total Columns**: ${dataset_info.column_count}
- **Memory Usage**: ${(dataset_info.memory_usage_bytes / (1024 * 1024)).toFixed(2)} MB
- **Missing Data**: ${missing_data_summary.missing_percent.toFixed(2)}% (${missing_data_summary.total_missing.toLocaleString()} out of ${missing_data_summary.total_cells.toLocaleString()} cells)

## Data Quality Assessment

### Missing Data Analysis
`;

    if (missing_data_summary.columns_with_missing.length > 0) {
      missing_data_summary.columns_with_missing.forEach((col: any) => {
        inferences += `- **${col.column}**: ${col.missing_count} missing values (${col.missing_percent.toFixed(2)}%)\n`;
      });
    } else {
      inferences += '- No missing data detected!\n';
    }

    inferences += `
### Detected ID Columns
`;
    const idColumns = columns.filter((col: any) => col.is_id_column);
    if (idColumns.length > 0) {
      idColumns.forEach((col: any) => {
        inferences += `- **${col.name}**: Unique identifier (${col.unique_count.toLocaleString()} unique values)\n`;
      });
    } else {
      inferences += '- No ID columns detected\n';
    }

    inferences += `
## Column Classification

### Numeric Features (${numeric_columns.length} columns)
`;
    numeric_columns.forEach((col: string, idx: number) => {
      const colInfo = columns.find((c: any) => c.name === col);
      const stats = statistics[col];
      if (stats) {
        inferences += `${idx + 1}. **${col}**: mean: ${stats.mean?.toFixed(2)}, std: ${stats.std?.toFixed(2)}, range: [${stats.min?.toFixed(2)}, ${stats.max?.toFixed(2)}]\n`;
      }
    });

    inferences += `
### Categorical Features (${categorical_columns.length} columns)
`;
    categorical_columns.forEach((col: string, idx: number) => {
      const colInfo = columns.find((c: any) => c.name === col);
      if (colInfo) {
        inferences += `${idx + 1}. **${col}**: ${colInfo.unique_count} unique values, ${colInfo.missing_count} missing\n`;
      }
    });

    if (correlations && correlations.pairs && correlations.pairs.length > 0) {
      inferences += `
## Correlation Analysis

### Strong Correlations (|r| > 0.7)
`;
      correlations.pairs.forEach((pair: any) => {
        inferences += `- **${pair.col1} ↔ ${pair.col2}**: r = ${pair.correlation.toFixed(3)}\n`;
      });
    }

    inferences += `
## Model Recommendations

Based on your data characteristics, here are the models available in our platform:

### For CLASSIFICATION Tasks:

#### 1. LightGBM (Light Gradient Boosting Machine) - HIGHLY RECOMMENDED ⭐⭐⭐
**Why this model is best for your data:**
- **Speed**: Extremely fast training, especially with ${dataset_info.row_count.toLocaleString()} rows
- **Memory efficient**: Uses less RAM than other gradient boosting methods
- **Handles your data perfectly**: Works great with ${numeric_columns.length} numeric and ${categorical_columns.length} categorical features
- **Missing data friendly**: Native support for ${missing_data_summary.missing_percent.toFixed(2)}% missing values (no preprocessing needed!)
- **Feature importance**: Tells you which columns matter most for predictions

**Technical Details:**
- Algorithm: Gradient boosting with leaf-wise tree growth
- Best for: Large datasets (>1000 rows), mixed data types
- Handles: Imbalanced classes, outliers, non-linear relationships

**Expected Performance:**
- Accuracy: 89-95%
- Training Time: Fast (seconds to minutes)
- ROC-AUC: 0.92-0.97

**When to use:** Perfect for production systems needing speed and accuracy

---

#### 2. XGBoost (Extreme Gradient Boosting) - HIGHLY RECOMMENDED ⭐⭐⭐
**Why this model is best for your data:**
- **Most powerful**: Often wins machine learning competitions
- **Robust**: Handles outliers and messy data extremely well
- **Versatile**: Works with any type of data (${numeric_columns.length} numeric, ${categorical_columns.length} categorical)
- **Regularization**: Built-in protection against overfitting
- **Missing values**: Learns best direction for missing data automatically

**Technical Details:**
- Algorithm: Gradient boosting with depth-wise tree growth
- Best for: When you need maximum accuracy, any dataset size
- Handles: Complex patterns, feature interactions, non-linear relationships

**Expected Performance:**
- Accuracy: 90-96%
- Training Time: Moderate (slightly slower than LightGBM)
- ROC-AUC: 0.93-0.98

**When to use:** When accuracy is top priority and training time is flexible

---

#### 3. Random Forest - RECOMMENDED ⭐⭐
**Why this model is good for your data:**
- **Easy to understand**: Simple concept (many decision trees voting)
- **No tuning needed**: Works well with default settings
- **Stable**: Less sensitive to parameter choices
- **Feature importance**: Shows which columns are most important
- **No scaling required**: Can use your data as-is

**Technical Details:**
- Algorithm: Ensemble of decision trees with bootstrap sampling
- Best for: Quick baseline, interpretable models, robust predictions
- Handles: ${dataset_info.row_count.toLocaleString()} rows very well

**Expected Performance:**
- Accuracy: 85-92%
- Training Time: Moderate
- ROC-AUC: 0.88-0.94

**When to use:** Need reliable results quickly without much tuning

---

#### 4. Logistic Regression - BASELINE ⭐
**Why this model might work for your data:**
- **Highly interpretable**: Easy to explain to stakeholders
- **Fast**: Trains in seconds even on large data
- **Simple**: Straightforward coefficients show feature impact
- **Good baseline**: Helps evaluate if complex models are needed

**Limitations for your data:**
- **Linear only**: Cannot capture complex patterns in ${numeric_columns.length} features
- **Needs preprocessing**: Requires scaling and encoding
- **Correlation issues**: May struggle with correlated features

**Technical Details:**
- Algorithm: Linear model with sigmoid activation
- Best for: Linear relationships, small datasets, highly interpretable needs
- Struggles with: Non-linear patterns, feature interactions

**Expected Performance:**
- Accuracy: 75-85%
- Training Time: Very fast (seconds)
- ROC-AUC: 0.80-0.88

**When to use:** Quick baseline or when model interpretability is critical

---

### For REGRESSION Tasks:

#### 1. LightGBM (Light Gradient Boosting Machine) - HIGHLY RECOMMENDED ⭐⭐⭐
**Why this model is best for your data:**
- **Fast and accurate**: Best combination of speed and performance
- **Handles ${numeric_columns.length} features** efficiently
- **Missing data support**: ${missing_data_summary.missing_percent.toFixed(2)}% missing values? No problem!
- **Continuous predictions**: Excellent for any range of target values

**Expected Performance:**
- R² Score: 0.87-0.94
- RMSE: Low to Very Low
- Training Time: Fast

---

#### 2. XGBoost (Extreme Gradient Boosting) - HIGHLY RECOMMENDED ⭐⭐⭐
**Why this model is best for your data:**
- **Maximum accuracy**: Industry standard for regression
- **Robust to outliers**: Won't be thrown off by extreme values
- **Feature interactions**: Automatically discovers relationships between ${numeric_columns.length} features
- **Regularization**: Prevents overfitting on ${dataset_info.row_count.toLocaleString()} samples

**Expected Performance:**
- R² Score: 0.88-0.95
- RMSE: Very Low
- Training Time: Moderate

---

#### 3. Random Forest - RECOMMENDED ⭐⭐
**Why this model is good for your data:**
- **Reliable**: Consistent performance across different datasets
- **No scaling needed**: Use raw feature values
- **Handles ${dataset_info.row_count.toLocaleString()} rows** well
- **Uncertainty estimates**: Can provide prediction confidence intervals

**Expected Performance:**
- R² Score: 0.82-0.91
- RMSE: Low
- Training Time: Moderate

---

#### 4. Linear Regression - BASELINE ⭐
**Why this model might work:**
- **Interpretable**: Clear coefficient for each feature
- **Fast**: Trains instantly
- **Simple**: Easy to understand and explain

**Limitations:**
- **Linear relationships only**: Assumes straight-line relationships
- **Sensitive to outliers**: Extreme values can skew predictions
- **Needs feature engineering**: May require creating interaction terms manually

**Expected Performance:**
- R² Score: 0.70-0.82 (if relationships are linear)
- RMSE: Moderate
- Training Time: Very fast

---

## Quick Decision Guide:

**🎯 For Maximum Accuracy:** Choose **XGBoost** or **LightGBM**
**⚡ For Speed + Accuracy:** Choose **LightGBM**
**🔍 For Interpretability:** Choose **Logistic/Linear Regression** or **Random Forest**
**🚀 For Quick Baseline:** Choose **Random Forest**
**📊 For Production Systems:** Choose **LightGBM** (fast inference)

## Feature Engineering Recommendations

### Handle Missing Data:
`;

    if (missing_data_summary.columns_with_missing.length > 0) {
      missing_data_summary.columns_with_missing.forEach((col: any) => {
        inferences += `- **${col.column}**: Consider imputation strategy (mean/median for numeric, mode for categorical)\n`;
      });
    }

    inferences += `
### Feature Scaling:
- Normalize/standardize numeric features for distance-based algorithms
- Not required for tree-based models (Random Forest, XGBoost)

### Categorical Encoding:
- Use one-hot encoding for ${categorical_columns.length} categorical features
- Consider target encoding for high-cardinality categories

## Training Configuration

### Recommended Split:
- **Train**: 70% (${Math.floor(dataset_info.row_count * 0.7).toLocaleString()} samples)
- **Validation**: 15% (${Math.floor(dataset_info.row_count * 0.15).toLocaleString()} samples)
- **Test**: 15% (${Math.floor(dataset_info.row_count * 0.15).toLocaleString()} samples)

### Cross-Validation:
- **Method**: 5-Fold Stratified K-Fold (for classification)
- **Metric**: ROC-AUC for classification, R² for regression
`;

    return inferences;
  };

  // Generate Model Config phase context
  const generateConfigContext = () => {
    let context = '# Model Configuration Phase\n\n';

    if (edaData) {
      const { dataset_info, numeric_columns, categorical_columns } = edaData;
      context += `## Dataset Summary\n`;
      context += `- Total Features: ${numeric_columns.length + categorical_columns.length}\n`;
      context += `- Numeric: ${numeric_columns.length}, Categorical: ${categorical_columns.length}\n`;
      context += `- Total Rows: ${dataset_info.row_count.toLocaleString()}\n\n`;
    }

    if (experimentConfig) {
      context += `## Current Configuration\n`;
      context += `- Task Type: ${experimentConfig.taskType || 'Not set'}\n`;
      context += `- Target Column: ${experimentConfig.targetColumn || 'Not selected'}\n`;
      context += `- Selected Features: ${experimentConfig.selectedFeatures?.length || 0} features\n`;
      if (experimentConfig.selectedFeatures?.length > 0) {
        context += `  - Features: ${experimentConfig.selectedFeatures.slice(0, 10).join(', ')}${experimentConfig.selectedFeatures.length > 10 ? '...' : ''}\n`;
      }
      context += `- Models Selected: ${experimentConfig.models?.length || 0}\n`;
      if (experimentConfig.models?.length > 0) {
        context += `  - Models: ${experimentConfig.models.map((m: any) => m.name).join(', ')}\n`;
      }
      context += `\n`;
    }

    context += `## Available Models\n`;
    context += `**Classification:** Logistic Regression, Random Forest, XGBoost, LightGBM\n`;
    context += `**Regression:** Linear Regression, Random Forest, XGBoost, LightGBM\n\n`;
    context += `**Recommendation:** For ${edaData?.dataset_info?.row_count > 10000 ? 'large' : 'medium'} datasets, LightGBM or XGBoost typically perform best.\n`;

    return context;
  };

  // Generate Training phase context
  const generateTrainingContext = () => {
    let context = '# Training Phase\n\n';

    if (trainingData && trainingData.length > 0) {
      context += `## Training Status\n`;
      context += `- Total Models Training: ${trainingData.length}\n\n`;

      trainingData.forEach((run: any, idx: number) => {
        context += `### Model ${idx + 1}: ${run.modelName || 'Unknown'}\n`;
        context += `- Status: ${run.status}\n`;
        if (run.metrics) {
          context += `- Current Metrics:\n`;
          Object.entries(run.metrics).forEach(([key, value]) => {
            context += `  - ${key}: ${typeof value === 'number' ? value.toFixed(4) : value}\n`;
          });
        }
        if (run.progress) {
          context += `- Progress: ${run.progress}%\n`;
        }
        context += `\n`;
      });
    } else {
      context += '## No Active Training\n';
      context += 'Training has not started yet or no training data available.\n\n';
    }

    context += `## Training Tips\n`;
    context += `- **Loss increasing?** May indicate learning rate too high or overfitting\n`;
    context += `- **Slow convergence?** Try increasing learning rate or checking feature scaling\n`;
    context += `- **Perfect training accuracy?** Watch for overfitting - check validation metrics\n`;

    return context;
  };

  // Generate Results phase context
  const generateResultsContext = () => {
    let context = '# Results Analysis\n\n';

    if (resultsData && resultsData.modelResults) {
      context += `## Model Performance Summary\n`;
      context += `- Total Models Trained: ${resultsData.modelResults.length}\n\n`;

      // Sort by primary metric (accuracy or R²)
      const sortedResults = [...resultsData.modelResults].sort((a, b) => {
        const metricA = a.metrics?.accuracy || a.metrics?.r2_score || 0;
        const metricB = b.metrics?.accuracy || b.metrics?.r2_score || 0;
        return metricB - metricA;
      });

      sortedResults.forEach((result: any, idx: number) => {
        context += `### ${idx + 1}. ${result.modelName}\n`;
        if (result.metrics) {
          Object.entries(result.metrics).forEach(([key, value]) => {
            context += `- ${key}: ${typeof value === 'number' ? value.toFixed(4) : value}\n`;
          });
        }
        context += `\n`;
      });

      if (resultsData.bestModel) {
        context += `## 🏆 Best Model: ${resultsData.bestModel.name}\n`;
        context += `This model achieved the highest performance on your validation set.\n\n`;
      }
    } else {
      context += '## No Results Available\n';
      context += 'Training needs to be completed to see results.\n\n';
    }

    context += `## Metrics Explained\n`;
    context += `- **Accuracy**: % of correct predictions (higher is better)\n`;
    context += `- **Precision**: Of predicted positives, how many are actually positive?\n`;
    context += `- **Recall**: Of actual positives, how many did we catch?\n`;
    context += `- **F1 Score**: Balance between precision and recall\n`;
    context += `- **ROC-AUC**: Model's ability to distinguish between classes (0.5 = random, 1.0 = perfect)\n`;

    return context;
  };

  // Generate Prediction phase context
  const generatePredictionContext = () => {
    let context = '# Prediction Phase\n\n';
    
    if (resultsData?.bestModel) {
      context += `## Selected Model: ${resultsData.bestModel.name}\n`;
      context += `This model will be used for predictions.\n\n`;
    }

    context += `## Making Predictions\n`;
    context += `- Upload new data or use sample data\n`;
    context += `- Ensure data has same features as training data\n`;
    context += `- Model will output predicted values with confidence scores\n\n`;

    context += `## Interpreting Predictions\n`;
    context += `- **Confidence Score**: Model's certainty (0-100%)\n`;
    context += `- **Low confidence?** Model is uncertain - verify input data quality\n`;
    context += `- **Unexpected results?** Check if input data matches training distribution\n`;

    return context;
  };

  const systemPrompt = `You are an expert Data Science Assistant designed to help BOTH technical and non-technical users understand their data and make informed decisions about machine learning models.

YOUR COMMUNICATION STYLE:
- Start with simple, clear explanations (for non-technical users)
- Then provide technical details (for technical users)
- Use analogies and real-world examples when helpful
- Avoid jargon unless explaining technical concepts
- Be encouraging and supportive

CRITICAL RULES:
1. Base ALL your answers EXCLUSIVELY on the context provided below.
2. If a question cannot be answered from the provided data, clearly state: "I don't have enough information in the current context to answer that question."
3. Never make up or assume information that isn't in the provided context.
4. Always explain WHY you recommend something, not just WHAT to do.
5. Keep responses concise (max 250 words) unless user asks for detailed explanation.

CURRENT PHASE: ${currentPhase.toUpperCase()}

YOUR PRIMARY TASKS:

**For Non-Technical Users:**
- Explain what the data shows in plain English
- Recommend models using simple language (e.g., "This model is like having 1000 experts vote on the answer")
- Explain expected results in business terms ("You can expect 90% accuracy, meaning 9 out of 10 predictions will be correct")
- Provide actionable next steps

**For Technical Users:**
- Provide detailed technical specifications
- Explain algorithms and mathematical concepts
- Discuss hyperparameters and optimization strategies
- Compare model architectures and computational complexity

**Questions You Should Handle:**

1. **Model Selection Questions:**
   - "Which model is best for my data?" → Recommend based on dataset characteristics with clear reasoning
   - "Why should I use XGBoost over Random Forest?" → Explain differences and when each excels
   - "What's the difference between LightGBM and XGBoost?" → Compare speed, accuracy, memory usage
   - "Is Logistic Regression good enough?" → Evaluate based on data complexity

2. **Data Quality Questions:**
   - "Do I have too much missing data?" → Assess missing data percentage and provide guidance
   - "Should I remove outliers?" → Advise based on model robustness
   - "Is my dataset too small?" → Evaluate dataset size for model requirements

3. **Feature Questions:**
   - "What features are most important?" → Reference correlations and feature types
   - "Should I create new features?" → Suggest based on relationships in data
   - "Can I use all columns?" → Identify ID columns to exclude

4. **Performance Questions:**
   - "What accuracy can I expect?" → Provide ranges based on data characteristics
   - "How long will training take?" → Estimate based on data size and model choice
   - "Will my model overfit?" → Assess based on data size and complexity

5. **Non-Technical Questions:**
   - "What does this data tell me?" → Summarize key insights in simple terms
   - "How do I know if my model is good?" → Explain metrics in plain language
   - "What should I do next?" → Provide step-by-step guidance
   - "Is this hard to do?" → Reassure and explain the process

6. **Comparison Questions:**
   - "LightGBM vs XGBoost - which one?" → Compare for their specific data
   - "Should I use ensemble models?" → Explain benefits for their use case
   - "Linear vs non-linear models?" → Assess data relationships

**AVAILABLE MODELS (Always mention these are the models in our platform):**

Classification: Logistic Regression, Random Forest, XGBoost, LightGBM
Regression: Linear Regression, Random Forest, XGBoost, LightGBM

**Model Recommendation Framework:**
1. **First choice (Best accuracy + speed)**: LightGBM or XGBoost
2. **Safe choice (Reliable, easy)**: Random Forest
3. **Baseline (Quick test)**: Logistic/Linear Regression
4. Always explain WHY each model suits their specific data

CONTEXT FOR CURRENT PHASE:
${generatePhaseContext()}

**Response Structure:**
1. **Direct Answer**: Start with clear answer to their question
2. **Simple Explanation**: Explain in non-technical terms
3. **Technical Details** (if relevant): Provide deeper technical context
4. **Recommendation**: Specific action they should take
5. **Why It Matters**: Connect to their goals

**Example Responses:**

User: "Which model should I use?"
You: "Based on your dataset characteristics (check the DATA ANALYSIS CONTEXT above for specific numbers), I recommend **LightGBM** as your primary choice. Here's why:

**Simple explanation:** Think of LightGBM as having thousands of simple decision trees working together to make predictions. It's like asking 1000 experts for their opinion and taking the most popular answer.

**Why it's best for YOUR data:**
- Handles your numeric and categorical features efficiently
- Naturally deals with missing data (no preprocessing needed!)
- Fast training on your dataset size
- Expected accuracy: 89-95%

**Technical details:** LightGBM uses leaf-wise tree growth with gradient boosting, providing faster training than level-wise approaches while maintaining high accuracy.

**Also consider:** XGBoost if maximum accuracy is critical and training time is less important."

Be conversational, helpful, and always tie recommendations back to the user's specific data characteristics shown in the DATA ANALYSIS CONTEXT above.`;

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Add empty assistant message that will be filled with streaming content
    const assistantMessageIndex = messages.length + 1;
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    try {
      const requestBody = {
        contents: [
          {
            role: 'user',
            parts: [{ text: systemPrompt }]
          },
          {
            role: 'model',
            parts: [{ text: 'Understood. I am ready to assist with data science questions based on the provided analysis context. I will only reference information from the given data and will clearly state when I cannot answer a question.' }]
          },
          ...messages.slice(1).map(msg => ({
            role: msg.role === 'user' ? 'user' : 'model',
            parts: [{ text: msg.content }]
          })),
          {
            role: 'user',
            parts: [{ text: input }]
          }
        ],
        generationConfig: {
          temperature: 0.7,
          topK: 40,
          topP: 0.95,
          maxOutputTokens: 1024,
        }
      };

      const response = await fetch(GEMINI_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const jsonStr = line.slice(6); // Remove 'data: ' prefix
                const data = JSON.parse(jsonStr);
                
                if (data.candidates && data.candidates[0]?.content?.parts?.[0]?.text) {
                  const newText = data.candidates[0].content.parts[0].text;
                  accumulatedText += newText;
                  
                  // Update the assistant message with streaming content
                  setMessages(prev => {
                    const updated = [...prev];
                    updated[assistantMessageIndex] = {
                      role: 'assistant',
                      content: accumulatedText
                    };
                    return updated;
                  });

                  // Add delay for slower, more readable streaming (30ms per chunk)
                  await new Promise(resolve => setTimeout(resolve, 30));
                }
              } catch (e) {
                // Skip invalid JSON lines
                continue;
              }
            }
          }
        }
      }
    } catch (error: any) {
      console.error('Chat error:', error);
      setMessages(prev => {
        const updated = [...prev];
        updated[assistantMessageIndex] = {
          role: 'assistant',
          content: '❌ Sorry, I encountered an error. Please try again or check the console for details.'
        };
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const buttonRect = buttonRef.current?.getBoundingClientRect();
    if (!buttonRect) return;

    setIsDragging(true);
    setDragStart({
      x: e.clientX - buttonRect.left,
      y: e.clientY - buttonRect.top
    });
  };

  const handleMouseMove = (e: MouseEvent) => {
    if (!isDragging) return;

    e.preventDefault();

    // Calculate new position based on cursor position
    const newX = e.clientX - dragStart.x;
    const newY = e.clientY - dragStart.y;

    // Constrain within viewport
    const maxX = window.innerWidth - 60;
    const maxY = window.innerHeight - 60;

    setPosition({
      x: Math.max(0, Math.min(newX, maxX)),
      y: Math.max(0, Math.min(newY, maxY))
    });
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, dragStart]);

  // Resize handlers
  const handleResizeMouseDown = (e: React.MouseEvent, direction: 'corner-br' | 'corner-bl' | 'corner-tr' | 'corner-tl' | 'bottom' | 'top' | 'right' | 'left') => {
    e.preventDefault();
    e.stopPropagation();
    
    setIsResizing(true);
    setResizeDirection(direction);
    setResizeStart({
      x: e.clientX,
      y: e.clientY,
      width: chatSize.width,
      height: chatSize.height
    });
  };

  const handleResizeMouseMove = (e: MouseEvent) => {
    if (!isResizing || !resizeDirection) return;

    e.preventDefault();

    const deltaX = e.clientX - resizeStart.x;
    const deltaY = e.clientY - resizeStart.y;

    // Apply constraints based on resize direction
    if (resizeDirection === 'corner-br') {
      // Bottom-right corner: increase width and height
      const newWidth = Math.max(320, Math.min(800, resizeStart.width + deltaX));
      const newHeight = Math.max(400, Math.min(800, resizeStart.height + deltaY));
      setChatSize({ width: newWidth, height: newHeight });
    } else if (resizeDirection === 'corner-bl') {
      // Bottom-left corner: decrease width from left, increase height
      const newWidth = Math.max(320, Math.min(800, resizeStart.width - deltaX));
      const newHeight = Math.max(400, Math.min(800, resizeStart.height + deltaY));
      setChatSize({ width: newWidth, height: newHeight });
    } else if (resizeDirection === 'corner-tr') {
      // Top-right corner: increase width, decrease height from top
      const newWidth = Math.max(320, Math.min(800, resizeStart.width + deltaX));
      const newHeight = Math.max(400, Math.min(800, resizeStart.height - deltaY));
      setChatSize({ width: newWidth, height: newHeight });
    } else if (resizeDirection === 'corner-tl') {
      // Top-left corner: decrease width from left, decrease height from top
      const newWidth = Math.max(320, Math.min(800, resizeStart.width - deltaX));
      const newHeight = Math.max(400, Math.min(800, resizeStart.height - deltaY));
      setChatSize({ width: newWidth, height: newHeight });
    } else if (resizeDirection === 'right') {
      // Resize width only (increase from right)
      const newWidth = Math.max(320, Math.min(800, resizeStart.width + deltaX));
      setChatSize({ width: newWidth, height: resizeStart.height });
    } else if (resizeDirection === 'left') {
      // Resize width only (decrease from left)
      const newWidth = Math.max(320, Math.min(800, resizeStart.width - deltaX));
      setChatSize({ width: newWidth, height: resizeStart.height });
    } else if (resizeDirection === 'bottom') {
      // Resize height only (increase from bottom)
      const newHeight = Math.max(400, Math.min(800, resizeStart.height + deltaY));
      setChatSize({ width: resizeStart.width, height: newHeight });
    } else if (resizeDirection === 'top') {
      // Resize height only (decrease from top)
      const newHeight = Math.max(400, Math.min(800, resizeStart.height - deltaY));
      setChatSize({ width: resizeStart.width, height: newHeight });
    }
  };

  const handleResizeMouseUp = () => {
    setIsResizing(false);
    setResizeDirection(null);
  };

  useEffect(() => {
    if (isResizing) {
      window.addEventListener('mousemove', handleResizeMouseMove);
      window.addEventListener('mouseup', handleResizeMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleResizeMouseMove);
        window.removeEventListener('mouseup', handleResizeMouseUp);
      };
    }
  }, [isResizing, resizeDirection, resizeStart]);

  if (!edaData && currentPhase === 'analysis') {
    return null; // Don't show assistant until EDA data is available in analysis phase
  }

  return (
    <>
      {/* Floating Button */}
      <div
        ref={buttonRef}
        className={`fixed z-50 ${isDragging ? 'cursor-grabbing' : 'cursor-grab'}`}
        style={{
          left: position.x === 0 && position.y === 0 ? 'auto' : `${position.x}px`,
          top: position.x === 0 && position.y === 0 ? 'auto' : `${position.y}px`,
          bottom: position.x === 0 && position.y === 0 ? '20px' : 'auto',
          right: position.x === 0 && position.y === 0 ? '20px' : 'auto',
        }}
        onMouseDown={handleMouseDown}
      >
        <Button
          onClick={() => !isDragging && setIsOpen(!isOpen)}
          className={`w-14 h-14 rounded-full shadow-lg gradient-primary transition-all glow-primary ${
            isDragging ? 'scale-110 cursor-grabbing' : 'hover:scale-105 cursor-grab'
          }`}
          size="icon"
        >
          <MessageCircle className="w-6 h-6" />
        </Button>
      </div>

      {/* Chat Window */}
      {isOpen && (
        <Card 
          ref={chatRef}
          className="fixed bottom-24 right-6 flex flex-col shadow-2xl z-50 border-primary/20 bg-card"
          style={{
            width: `${chatSize.width}px`,
            height: `${chatSize.height}px`,
          }}
        >
          {/* Header */}
          <div className="gradient-primary text-primary-foreground p-4 rounded-t-lg flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-lg">🤖 Data Science Assistant</h3>
              {/* <p className="text-xs opacity-90">Powered by Gemini 2.0 Flash</p> */}
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsOpen(false)}
              className="text-primary-foreground hover:bg-primary-foreground/20 h-8 w-8"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-background">
            {messages.map((message, idx) => (
              <div
                key={idx}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] px-4 py-2 rounded-2xl ${
                    message.role === 'user'
                      ? 'gradient-primary text-primary-foreground rounded-br-sm'
                      : 'bg-card border border-border text-card-foreground rounded-bl-sm'
                  }`}
                >
                  {message.role === 'user' ? (
                    <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                  ) : (
                    <div className="text-sm prose prose-invert prose-sm max-w-none
                      prose-headings:text-primary prose-headings:font-semibold prose-headings:mb-2 prose-headings:mt-3
                      prose-h1:text-base prose-h2:text-sm prose-h3:text-sm
                      prose-p:my-2 prose-p:leading-relaxed prose-p:text-card-foreground
                      prose-strong:text-primary prose-strong:font-semibold
                      prose-ul:my-2 prose-ul:list-disc prose-ul:pl-4
                      prose-ol:my-2 prose-ol:list-decimal prose-ol:pl-4
                      prose-li:my-1 prose-li:text-card-foreground
                      prose-code:text-primary prose-code:bg-primary/10 prose-code:px-1 prose-code:py-0.5 prose-code:rounded
                      prose-hr:my-3 prose-hr:border-border
                      prose-blockquote:border-l-primary prose-blockquote:italic prose-blockquote:pl-4">
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-card border border-border px-4 py-2 rounded-2xl rounded-bl-sm">
                  <Loader2 className="w-4 h-4 animate-spin text-primary" />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 bg-card border-t border-border">
            <div className="flex gap-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !isLoading && sendMessage()}
                placeholder={`Ask about ${currentPhase}...`}
                className="flex-1 rounded-full bg-input border-border focus:ring-primary"
                disabled={isLoading}
              />
              <Button
                onClick={sendMessage}
                disabled={isLoading || !input.trim()}
                size="icon"
                className="rounded-full gradient-primary hover:scale-105 transition-transform"
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Resize Handles - Corners */}
          {/* Bottom-Right Corner */}
          <div
            onMouseDown={(e) => handleResizeMouseDown(e, 'corner-br')}
            className="absolute bottom-0 right-0 w-6 h-6 cursor-nwse-resize group z-10"
            style={{ touchAction: 'none' }}
          >
            <div className="absolute bottom-1 right-1 w-4 h-4 border-r-2 border-b-2 border-primary/40 group-hover:border-primary transition-colors" />
          </div>

          {/* Bottom-Left Corner */}
          <div
            onMouseDown={(e) => handleResizeMouseDown(e, 'corner-bl')}
            className="absolute bottom-0 left-0 w-6 h-6 cursor-nesw-resize group z-10"
            style={{ touchAction: 'none' }}
          >
            <div className="absolute bottom-1 left-1 w-4 h-4 border-l-2 border-b-2 border-primary/40 group-hover:border-primary transition-colors" />
          </div>

          {/* Top-Right Corner */}
          <div
            onMouseDown={(e) => handleResizeMouseDown(e, 'corner-tr')}
            className="absolute top-0 right-0 w-6 h-6 cursor-nesw-resize group z-10"
            style={{ touchAction: 'none' }}
          >
            <div className="absolute top-1 right-1 w-4 h-4 border-r-2 border-t-2 border-primary/40 group-hover:border-primary transition-colors" />
          </div>

          {/* Top-Left Corner */}
          <div
            onMouseDown={(e) => handleResizeMouseDown(e, 'corner-tl')}
            className="absolute top-0 left-0 w-6 h-6 cursor-nwse-resize group z-10"
            style={{ touchAction: 'none' }}
          >
            <div className="absolute top-1 left-1 w-4 h-4 border-l-2 border-t-2 border-primary/40 group-hover:border-primary transition-colors" />
          </div>

          {/* Resize Handles - Edges */}
          {/* Bottom Edge */}
          <div
            onMouseDown={(e) => handleResizeMouseDown(e, 'bottom')}
            className="absolute bottom-0 left-6 right-6 h-3 cursor-ns-resize hover:bg-primary/10 transition-colors z-[5]"
            style={{ touchAction: 'none' }}
          />

          {/* Top Edge */}
          <div
            onMouseDown={(e) => handleResizeMouseDown(e, 'top')}
            className="absolute top-0 left-6 right-6 h-3 cursor-ns-resize hover:bg-primary/10 transition-colors z-[5]"
            style={{ touchAction: 'none' }}
          />

          {/* Right Edge */}
          <div
            onMouseDown={(e) => handleResizeMouseDown(e, 'right')}
            className="absolute top-6 bottom-6 right-0 w-3 cursor-ew-resize hover:bg-primary/10 transition-colors z-[5]"
            style={{ touchAction: 'none' }}
          />

          {/* Left Edge */}
          <div
            onMouseDown={(e) => handleResizeMouseDown(e, 'left')}
            className="absolute top-6 bottom-6 left-0 w-3 cursor-ew-resize hover:bg-primary/10 transition-colors z-[5]"
            style={{ touchAction: 'none' }}
          />
        </Card>
      )}
    </>
  );
};
