import re
import random
import json
from typing import List, Dict, Tuple
from config import Config

class OriginalLinkedInDataGenerator:
    def __init__(self):
        self.config = Config()
        
        self.vocabulary = {
            'leadership_terms': [
                'visionary', 'inspiring', 'empowering', 'guiding', 'mentoring',
                'strategic', 'decisive', 'collaborative', 'innovative', 'adaptive'
            ],
            'career_terms': [
                'growth', 'development', 'advancement', 'opportunity', 'achievement',
                'milestone', 'journey', 'transition', 'success', 'progress'
            ],
            'business_terms': [
                'strategy', 'innovation', 'transformation', 'optimization', 'efficiency',
                'productivity', 'performance', 'results', 'impact', 'value'
            ],
            'soft_skills': [
                'communication', 'collaboration', 'adaptability', 'creativity', 'resilience',
                'empathy', 'integrity', 'accountability', 'initiative', 'teamwork'
            ],
            'industries': [
                'technology', 'finance', 'healthcare', 'education', 'manufacturing',
                'retail', 'consulting', 'marketing', 'sales', 'operations'
            ],
            'emotions': [
                'excited', 'grateful', 'proud', 'inspired', 'motivated',
                'honored', 'thrilled', 'passionate', 'determined', 'optimistic'
            ],
            'technologies': [
                'AI', 'machine learning', 'blockchain', 'cloud computing', 'automation',
                'data analytics', 'IoT', 'cybersecurity', 'digital transformation', 'mobile technology'
            ],
            'trends': [
                'remote work', 'hybrid models', 'digital transformation', 'sustainability',
                'personalization', 'automation', 'data-driven decisions', 'agile methodologies'
            ]
        }
        
        self.templates = {
            'career_advice': [
                "After {years} years in {industry}, I've learned that {insight}. Here's my advice: {advice}",
                "The best career advice I ever received: {advice}. It changed how I approach {topic}.",
                "{number} lessons I wish I knew when starting my career in {industry}: {lessons}",
                "Career growth isn't just about {skill1}, it's about {skill2}. Here's why: {explanation}"
            ],
            'industry_insights': [
                "The {industry} industry is evolving rapidly. Here are {number} trends I'm watching: {trends}",
                "What {year} taught us about {industry}: {insights}",
                "Why {technology} will reshape {industry} in the next {years} years: {predictions}",
                "The future of {industry} depends on {factor}. Here's my perspective: {analysis}"
            ],
            'leadership': [
                "Great leaders don't just {action1}, they {action2}. Here's what I've learned: {lessons}",
                "Leadership lesson from {situation}: {insight}",
                "The difference between management and leadership? {explanation}",
                "{number} qualities that separate good leaders from great ones: {qualities}"
            ],
            'entrepreneurship': [
                "Starting my own business taught me {lesson}. Here's what every entrepreneur should know: {advice}",
                "The biggest challenge in entrepreneurship isn't {challenge1}, it's {challenge2}. Here's why: {explanation}",
                "{number} mistakes I made as a first-time entrepreneur: {mistakes}",
                "Building a startup is {adjective}. Here's what kept me going: {motivation}"
            ],
            'professional_development': [
                "Investing in {skill} has been game-changing for my career. Here's why: {benefits}",
                "{number} skills every {role} should develop: {skills}",
                "The most valuable skill I developed this year: {skill}. Here's how: {method}",
                "Why continuous learning is non-negotiable in {industry}: {reasons}"
            ],
            'technology_trends': [
                "{technology} is transforming how we work. Here are {number} ways it's impacting {industry}: {impacts}",
                "The future of {technology}: {prediction}. Here's what businesses need to know: {advice}",
                "Why {technology} adoption is accelerating in {year}: {reasons}",
                "{number} {technology} trends every {role} should watch: {trends}"
            ],
            'personal_branding': [
                "Your personal brand is your {asset}. Here's how to build it: {steps}",
                "Personal branding isn't about {misconception}, it's about {reality}. Here's why: {explanation}",
                "{number} ways to strengthen your professional presence: {methods}",
                "The biggest personal branding mistake I see: {mistake}. Here's how to avoid it: {solution}"
            ],
            'networking': [
                "Networking changed my career trajectory. Here's how: {story}",
                "The best networking advice: {advice}. It's not about {wrong_approach}, it's about {right_approach}",
                "{number} networking mistakes that hurt your career: {mistakes}",
                "Building genuine professional relationships requires {requirement}. Here's my approach: {method}"
            ],
            'innovation': [
                "Innovation doesn't happen by accident. It requires {requirement}. Here's how we foster it: {method}",
                "The biggest barrier to innovation? {barrier}. Here's how to overcome it: {solution}",
                "{number} ways to cultivate an innovative mindset: {methods}",
                "True innovation comes from {source}. Here's what I've learned: {insight}"
            ],
            'workplace_culture': [
                "Company culture isn't just {surface_thing}, it's {deep_thing}. Here's why it matters: {importance}",
                "Building a positive workplace culture requires {requirement}. Here's our approach: {method}",
                "{number} signs of a toxic workplace culture: {signs}",
                "The culture change that transformed our team: {change}. Here's what we learned: {lesson}"
            ],
            'personal_story': [
                "Last {timeframe}, {event}. It reminded me that {lesson}.",
                "A {adjective} moment in my career: {story}. The takeaway? {insight}",
                "I'll never forget when {event}. It taught me {lesson}.",
                "From {challenge} to {success}: {story}"
            ]
        }
        
        self.fillers = {
            'years': ['3', '5', '10', '15', '20', 'several', 'many'],
            'number': ['3', '5', '7', '10'],
            'timeframe': ['week', 'month', 'quarter', 'year'],
            'adjective': ['defining', 'pivotal', 'transformative', 'memorable', 'significant'],
            'year': ['2024', '2025', 'this year', 'last year'],
            'role': ['professional', 'leader', 'manager', 'entrepreneur', 'developer', 'marketer'],
            'asset': ['greatest asset', 'most valuable tool', 'competitive advantage', 'key differentiator'],
            'requirement': ['intentional effort', 'systematic approach', 'consistent practice', 'strategic thinking']
        }
    
    def _select_random(self, items: List[str]) -> str:
        return random.choice(items)
    
    def _generate_content_filler(self, placeholder: str) -> str:
        if placeholder in self.fillers:
            return self._select_random(self.fillers[placeholder])
        
        if placeholder in self.vocabulary:
            return self._select_random(self.vocabulary[placeholder])
        
        if 'insight' in placeholder or 'lesson' in placeholder:
            return self._generate_insight()
        elif 'advice' in placeholder:
            return self._generate_advice()
        elif 'trend' in placeholder or 'trends' in placeholder:
            return self._generate_trends()
        elif 'skill' in placeholder:
            return self._select_random(self.vocabulary['soft_skills'])
        elif 'industry' in placeholder:
            return self._select_random(self.vocabulary['industries'])
        elif 'technology' in placeholder:
            return self._select_random(self.vocabulary['technologies'])
        elif 'challenge' in placeholder:
            return self._generate_challenge()
        elif 'solution' in placeholder:
            return self._generate_solution()
        elif 'method' in placeholder or 'approach' in placeholder:
            return self._generate_method()
        elif 'explanation' in placeholder:
            return self._generate_explanation()
        elif 'prediction' in placeholder:
            return self._generate_prediction()
        elif 'impact' in placeholder or 'impacts' in placeholder:
            return self._generate_impacts()
        elif 'mistake' in placeholder or 'mistakes' in placeholder:
            return self._generate_mistakes()
        elif 'benefit' in placeholder or 'benefits' in placeholder:
            return self._generate_benefits()
        elif 'step' in placeholder or 'steps' in placeholder:
            return self._generate_steps()
        elif 'sign' in placeholder or 'signs' in placeholder:
            return self._generate_signs()
        elif 'change' in placeholder:
            return self._generate_change()
        elif 'story' in placeholder:
            return self._generate_story()
        elif 'event' in placeholder:
            return self._generate_event()
        elif 'success' in placeholder:
            return self._generate_success()
        elif 'topic' in placeholder:
            return self._select_random(['networking', 'leadership', 'career growth', 'innovation', 'teamwork'])
        elif 'action' in placeholder:
            return self._select_random(['delegate', 'communicate', 'inspire', 'strategize', 'collaborate'])
        elif 'situation' in placeholder:
            return self._select_random(['a crisis', 'team conflict', 'major project', 'difficult decision'])
        elif 'factor' in placeholder:
            return self._select_random(['innovation', 'talent', 'technology', 'customer focus', 'adaptability'])
        elif 'barrier' in placeholder:
            return self._select_random(['fear of failure', 'resistance to change', 'lack of resources', 'rigid thinking'])
        elif 'source' in placeholder:
            return self._select_random(['collaboration', 'diverse perspectives', 'continuous learning', 'customer feedback'])
        elif 'misconception' in placeholder:
            return self._select_random(['self-promotion', 'showing off', 'being fake', 'networking for gain'])
        elif 'reality' in placeholder:
            return self._select_random(['authenticity', 'value creation', 'genuine relationships', 'consistent presence'])
        elif 'wrong_approach' in placeholder:
            return self._select_random(['collecting contacts', 'transactional relationships', 'self-serving motives'])
        elif 'right_approach' in placeholder:
            return self._select_random(['building relationships', 'providing value', 'genuine connections'])
        elif 'surface_thing' in placeholder:
            return self._select_random(['perks and benefits', 'office design', 'company events'])
        elif 'deep_thing' in placeholder:
            return self._select_random(['shared values', 'psychological safety', 'growth mindset', 'mutual respect'])
        elif 'importance' in placeholder:
            return self._select_random(['it drives engagement', 'it attracts talent', 'it improves performance'])
        
        return "professional growth"
    
    def _generate_insight(self) -> str:
        insights = [
            "success comes from consistent daily actions",
            "networking is about giving before receiving", 
            "failure is the best teacher for growth",
            "authenticity builds stronger relationships",
            "continuous learning is the key to relevance",
            "leadership is about serving others",
            "innovation requires embracing uncertainty",
            "culture eats strategy for breakfast"
        ]
        return self._select_random(insights)
    
    def _generate_advice(self) -> str:
        advice = [
            "focus on building genuine relationships",
            "always be learning and adapting",
            "take calculated risks for growth",
            "invest in your personal brand",
            "seek mentorship and feedback",
            "prioritize value creation over self-promotion",
            "embrace failure as learning opportunity",
            "build diverse professional networks"
        ]
        return self._select_random(advice)
    
    def _generate_trends(self) -> str:
        trends = [
            "AI automation transforming workflows",
            "remote work becoming the new normal",
            "sustainability driving business decisions", 
            "data-driven decision making",
            "personalized customer experiences",
            "agile methodologies gaining adoption",
            "digital transformation accelerating",
            "employee well-being prioritization"
        ]
        return self._select_random(trends)
    
    def _generate_challenge(self) -> str:
        challenges = [
            "finding funding",
            "building the right team", 
            "market validation",
            "scaling operations",
            "managing cash flow",
            "staying competitive"
        ]
        return self._select_random(challenges)
    
    def _generate_solution(self) -> str:
        solutions = [
            "start with small experiments",
            "focus on customer feedback",
            "build strong partnerships",
            "invest in team development",
            "embrace data-driven decisions",
            "maintain clear communication"
        ]
        return self._select_random(solutions)
    
    def _generate_method(self) -> str:
        methods = [
            "regular team check-ins",
            "transparent communication",
            "celebrating small wins",
            "encouraging experimentation",
            "providing growth opportunities",
            "fostering psychological safety"
        ]
        return self._select_random(methods)
    
    def _generate_explanation(self) -> str:
        explanations = [
            "it builds trust and credibility",
            "it creates lasting value",
            "it drives sustainable growth",
            "it improves team performance",
            "it enhances customer satisfaction",
            "it fosters innovation"
        ]
        return self._select_random(explanations)
    
    def _generate_prediction(self) -> str:
        predictions = [
            "will revolutionize how we work",
            "will create new opportunities",
            "will require new skill sets",
            "will transform customer expectations",
            "will reshape entire industries",
            "will drive competitive advantage"
        ]
        return self._select_random(predictions)
    
    def _generate_impacts(self) -> str:
        impacts = [
            "streamlined processes",
            "improved decision making",
            "enhanced customer experience",
            "increased productivity",
            "reduced operational costs",
            "better collaboration"
        ]
        return self._select_random(impacts)
    
    def _generate_mistakes(self) -> str:
        mistakes = [
            "not validating assumptions early",
            "trying to do everything alone",
            "ignoring customer feedback",
            "scaling too quickly",
            "neglecting team culture",
            "avoiding difficult conversations"
        ]
        return self._select_random(mistakes)
    
    def _generate_benefits(self) -> str:
        benefits = [
            "improved problem-solving abilities",
            "increased career opportunities",
            "enhanced leadership capabilities",
            "better strategic thinking",
            "stronger professional network",
            "greater industry credibility"
        ]
        return self._select_random(benefits)
    
    def _generate_steps(self) -> str:
        steps = [
            "define your unique value proposition",
            "consistently share valuable insights",
            "engage authentically with others",
            "build a strong online presence",
            "seek speaking opportunities",
            "mentor others in your field"
        ]
        return self._select_random(steps)
    
    def _generate_signs(self) -> str:
        signs = [
            "high employee turnover",
            "lack of open communication",
            "resistance to change",
            "blame culture",
            "micromanagement",
            "no growth opportunities"
        ]
        return self._select_random(signs)
    
    def _generate_change(self) -> str:
        changes = [
            "implementing regular feedback sessions",
            "encouraging open communication",
            "recognizing team achievements",
            "providing learning opportunities",
            "promoting work-life balance",
            "fostering collaboration"
        ]
        return self._select_random(changes)
    
    def _generate_story(self) -> str:
        stories = [
            "I met a mentor who changed my perspective",
            "our team overcame a major challenge",
            "a failed project taught me valuable lessons",
            "I took a risk that paid off",
            "a customer's feedback transformed our approach",
            "I learned the importance of listening"
        ]
        return self._select_random(stories)
    
    def _generate_event(self) -> str:
        events = [
            "I spoke at a conference",
            "our project was recognized",
            "I received unexpected feedback",
            "a colleague shared their story",
            "I faced a difficult decision",
            "we launched a new initiative"
        ]
        return self._select_random(events)
    
    def _generate_success(self) -> str:
        successes = [
            "building a thriving team",
            "launching a successful product",
            "achieving work-life balance",
            "becoming a trusted leader",
            "creating lasting impact",
            "inspiring others to grow"
        ]
        return self._select_random(successes)
    
    def _fill_template(self, template: str) -> str:        
        placeholders = re.findall(r'\{(\w+)\}', template)
        
        filled_template = template
        for placeholder in placeholders:
            content = self._generate_content_filler(placeholder)
            filled_template = filled_template.replace(f'{{{placeholder}}}', content)
        
        return filled_template
    
    def _add_engagement_elements(self, post: str) -> str:
        hashtags = [
            '#Leadership', '#CareerGrowth', '#ProfessionalDevelopment', 
            '#Innovation', '#Success', '#Networking', '#Mindset',
            '#Growth', '#Inspiration', '#Business', '#Technology',
            '#Entrepreneurship', '#WorkplaceCulture', '#PersonalBranding'
        ]
        
        selected_hashtags = random.sample(hashtags, random.randint(2, 3))
        post += '\n\n' + ' '.join(selected_hashtags)
        
        if random.random() < 0.3:
            prompts = [
                "\n\nWhat's your experience with this?",
                "\n\nThoughts? Share in the comments below.",
                "\n\nWhat would you add to this list?",
                "\n\nHave you experienced something similar?",
                "\n\nHow do you approach this challenge?",
                "\n\nWhat's worked for you?"
            ]
            post += self._select_random(prompts)
        
        return post
    
    def generate_post(self, theme: str = None) -> Tuple[str, str]:
        if not theme:
            theme = self._select_random(list(self.templates.keys()))
        
        if theme not in self.templates:
            available_themes = list(self.templates.keys())
            theme = self._select_random(available_themes)
            print(f"Warning: Theme not found, using '{theme}' instead")
        
        template = self._select_random(self.templates[theme])
        post = self._fill_template(template)
        post = self._add_engagement_elements(post)
        
        return post, theme
    
    def generate_dataset(self, num_posts: int) -> List[Dict]:
        print(f"Generating {num_posts} LinkedIn posts...")
        
        dataset = []
        theme_distribution = {theme: 0 for theme in self.config.THEMES}
        
        available_themes = list(self.templates.keys())
        
        for i in range(num_posts):
            theme = self._select_random(available_themes)
            post, actual_theme = self.generate_post(theme)
            
            dataset.append({
                'id': i,
                'post': post,
                'theme': actual_theme,
                'length': len(post),
                'word_count': len(post.split())
            })
            
            if actual_theme in theme_distribution:
                theme_distribution[actual_theme] += 1
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1} posts...")
        
        print("Dataset generation complete!")
        print("Theme distribution:", theme_distribution)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> List[Dict]:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)