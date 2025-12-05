import os

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task






@CrewBase
class HolisticInterviewEvaluatorWithReferenceAnswersCrew:
    """HolisticInterviewEvaluatorWithReferenceAnswers crew"""

    
    @agent
    def holistic_interview_evaluator(self) -> Agent:
        
        return Agent(
            config=self.agents_config["holistic_interview_evaluator"],
            
            
            tools=[],
            reasoning=True,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="groq/llama-3.3-70b-versatile",
                temperature=0.2,
                api_key=os.getenv("GROQ_API_KEY"),
            ),
            
        )
    
    @agent
    def synthesizer(self) -> Agent:
        
        return Agent(
            config=self.agents_config["synthesizer"],
            
            
            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="groq/llama-3.1-8b-instant",
                temperature=0.7,
                api_key=os.getenv("GROQ_API_KEY"),
            ),
            
        )
    
    @agent
    def output_controller(self) -> Agent:
        
        return Agent(
            config=self.agents_config["output_controller"],
            
            
            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="groq/llama-3.3-70b-versatile",
                temperature=0.2,
                api_key=os.getenv("GROQ_API_KEY"),
            ),
            
        )
    
    @agent
    def reference_answer_generator(self) -> Agent:
        
        return Agent(
            config=self.agents_config["reference_answer_generator"],
            
            
            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="groq/llama-3.3-70b-versatile",
                temperature=0.2,
                api_key=os.getenv("GROQ_API_KEY"),
            ),
            
        )
    

    
    @task
    def generate_expected_answers(self) -> Task:
        return Task(
            config=self.tasks_config["generate_expected_answers"],
            markdown=False,
            
            
        )
    
    @task
    def holistic_interview_evaluation(self) -> Task:
        return Task(
            config=self.tasks_config["holistic_interview_evaluation"],
            markdown=False,
            
            
        )
    
    @task
    def synthesis_and_development_plan(self) -> Task:
        return Task(
            config=self.tasks_config["synthesis_and_development_plan"],
            markdown=False,
            
            
        )
    
    @task
    def final_output_assembly(self) -> Task:
        return Task(
            config=self.tasks_config["final_output_assembly"],
            markdown=False,
            
            
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the HolisticInterviewEvaluatorWithReferenceAnswers crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )

    def _load_response_format(self, name):
        with open(os.path.join(self.base_directory, "config", f"{name}.json")) as f:
            json_schema = json.loads(f.read())

        return SchemaConverter.build(json_schema)
