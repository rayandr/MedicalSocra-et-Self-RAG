import openai
import re
import os
from dotenv import load_dotenv
import time  # Import time for tracking execution

load_dotenv()

OPENAI_API_KEY_STAGE = os.getenv("OPENAI_API_KEY_STAGE")

class Crit:

    def __init__(self):
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY_STAGE)
        print("\nInitialized Crit class with OpenAI client.")
        # Track visited claims to prevent infinite recursion
        self.visited_claims = set()
        # Track start time to limit total processing time
        self.start_time = time.time()
        # Set max execution time (seconds)
        # self.max_execution_time = 120
        self.max_execution_time = 180
        

    def extract_claim(self, document):
        # Check if we've exceeded max execution time
        if time.time() - self.start_time > self.max_execution_time:
            print("\nExceeded maximum execution time, returning early.")
            return "Execution time limit reached"
            
        print("\nExtracting claim from document.")
        # Use a shorter text if the document is too long
        if len(document) > 1000:
            document = document[:1000] + "..."
            
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": "Extract the main claim from this document. Keep it concise, under 50 words."
            }, {
                "role": "user",
                "content": document
            }]
        )
        claim = response.choices[0].message.content.strip()
        print(f"\nExtracted claim: {claim}")
        return claim


    def compute_final_score(self, val_scores, counter_scores):
        """Compute final weighted score Γ = ∑ γ × θ"""
        print(f"\nComputing final score from validation scores and counter scores")
        
        # Ensure we have valid scores
        if not val_scores:
            return 5.0  # Default mid-range score
            
        # Calculate supporting scores - simpler implementation
        supporting_score = sum(gamma * theta for gamma, theta in val_scores) / sum(theta for _, theta in val_scores) if sum(theta for _, theta in val_scores) > 0 else 0
        
        # Calculate counter scores impact
        counter_score = 0
        if counter_scores:
            counter_sum = sum(gamma * theta for gamma, theta in counter_scores)
            counter_weight = sum(theta for _, theta in counter_scores)
            if counter_weight > 0:
                counter_score = counter_sum / (2 * counter_weight)  # Divide by 2 to reduce counter impact
        
        # Calculate final score (supporting - counter)
        final_score = supporting_score - counter_score
        
        # Normalize to 0-10 range
        final_score = max(0, min(10, final_score))
        
        print(f"\nCalculated final score: {final_score}")
        return round(final_score, 2)


    def crit(self, document, depth=0, max_depth=2, claim_id=None):
        """Main CRIT function with improved handling to prevent infinite recursion"""
        # Check if we're exceeding time limit to avoid long-running processes
        if time.time() - self.start_time > self.max_execution_time:
            print("\nExceeded maximum execution time, returning early.")
            return 5.0  # Return a neutral score
            
        # Base case: Prevent excessive recursion
        if depth > max_depth:
            print(f"\nReached maximum recursion depth ({max_depth})")
            return 5.0  # Return a neutral score
            
        print(f"\n{'='*20} STARTING CRIT ANALYSIS (DEPTH {depth}) {'='*20}")
        
        # Step 1: Extract claim
        claim = self.extract_claim(document)
        
        # Create a fingerprint for the claim to handle similar claims
        claim_fingerprint = claim[:100].lower()
        
        # Check if we've seen this claim or something very similar
        if claim_fingerprint in self.visited_claims:
            print(f"\nAlready visited similar claim, stopping recursion")
            return 5.0  # Return a neutral score
            
        # Mark this claim as visited
        self.visited_claims.add(claim_fingerprint)
        
        # Step 2: Evaluate at most two supporting reasons to limit API calls
        supporting_reasons = self.find_supporting_reasons(claim, document)[:2]
        if not supporting_reasons:
            print("\nNo supporting reasons found.")
            return 5.0  # Return a neutral score
            
        validation_scores = []
        for reason in supporting_reasons:
            # Skip if we're running out of time
            if time.time() - self.start_time > self.max_execution_time:
                break
                
            # print(f"Evaluating supporting reason: '{reason[:50]}...'")
            print(f"\nEvaluating supporting reason: '{reason}'")
            gamma, theta = self.validate_argument(reason, claim)
            validation_scores.append((gamma, theta))
            
        # Step 3: Get up to two counter-reasons to limit API calls
        rival_reasons = self.find_counterarguments(claim)[:2]
        rival_scores = []
        
        for rival in rival_reasons:
            # Skip if we're running out of time
            if time.time() - self.start_time > self.max_execution_time:
                break
                
            # print(f"Evaluating rival reason: '{rival[:50]}...'")
            print(f"\nEvaluating rival reason: '{rival}'")
            gamma_rival, theta_rival = self.validate_argument(rival, claim)
            rival_scores.append((gamma_rival, theta_rival))
            
        # Step 4: Compute final score
        final_score = self.compute_final_score(validation_scores, rival_scores)
        print(f"\n{'*'*20} FINAL CRIT SCORE: {final_score} {'*'*20}\n")
        
        # Remove claim from visited set to allow for future analysis
        self.visited_claims.remove(claim_fingerprint)
        
        return final_score
        

    def find_supporting_reasons(self, claim, doc):
        print(f"\nFinding supporting reasons for claim")
        prompt = f"""Identify 2-3 supporting reasons from this document that back up the following claim: 
                  Claim: {claim}
                  Document: {doc[:1000]}...  # Limiting text length
                  List each reason as a bullet point."""
      
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        reasons = self.extract_bullets(response.choices[0].message.content)
        print(f"\nFound {len(reasons)} supporting reasons")
        return reasons
    

    def find_counterarguments(self, claim):
        print(f"\nFinding counterarguments for claim")
        prompt = f"""List 2-3 potential counterarguments or rival claims to the following claim: 
                  Claim: {claim}
                  Return them as bullet points."""
      
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        counter_args = self.extract_bullets(response.choices[0].message.content)
        print(f"\nFound {len(counter_args)} counterarguments")
        return counter_args
    

    def validate_argument(self, reason, claim):
        print(f"\nValidating argument against claim")
        prompt = f"""Rate how well this reason supports the claim on two dimensions:
                1. γ (gamma): Argument validity (1-10)
                2. θ (theta): Source credibility (1-10)

                Claim: {claim}
                Reason: {reason}
                
                Return ONLY two scores as: "Gamma: X, Theta: Y"."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        gamma, theta = self.parse_scores(response.choices[0].message.content)
        print(f"Validation scores - γ: {gamma}, θ: {theta}")
        return gamma, theta
    
    def parse_scores(self, text):
        """Parse scores with better error handling"""
        numbers = re.findall(r'\b\d+\b', text)
        valid_numbers = [int(n) for n in numbers if 1 <= int(n) <= 10]
        
        if len(valid_numbers) >= 2:
            return valid_numbers[0], valid_numbers[1]
        
        print(f"\nWarning: Couldn't extract valid scores from: {text}")
        return 5, 5  # Default neutral values
    

    def extract_bullets(self, text):
        lines = text.strip().split("\n")
        return [re.sub(r'^[-*•\d.\)\s]+', '', line).strip() for line in lines if line.strip()]

# Example usage
# crit = Crit()
# doc = """
# Climate change is a significant challenge facing our world. Rising temperatures are affecting weather patterns 
# and leading to more extreme climate events. 
# """
# score = crit.crit(doc)
# print("CRIT Score:", score)



# crit = Crit()

# llm_plus_doc = "Universal Basic Income helps reduce poverty by giving people money unconditionally."
# llm_minus_doc = "Universal Basic Income can reduce the motivation to work and increase government debt."


# print("\nEvaluating LLM+ Claim:")
# score_plus = crit.crit(llm_plus_doc)

# print("\nEvaluating LLM− Claim:")
# score_minus = crit.crit(llm_minus_doc)

# print(f"\nLLM+ CRIT Score: {score_plus}")
# print(f"LLM− CRIT Score: {score_minus}")
