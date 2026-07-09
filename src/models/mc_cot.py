class MCCoTReasoner:
    def __init__(self, generator_model):
        self.generator = generator_model
        self.modules = {
            "anatomy": "You are an anatomy expert. Analyze the structures visible.",
            "pathology": "You are a pathology expert. Identify abnormal findings.",
            "clinical_significance": "You are a clinical expert. Explain the clinical meaning."
        }

        def reason(self, image, question):
            context_chunks = []
            for mod_name, system_prompt in self.modules.items():
                prompt = f"{system_prompt}\nQuestion: {question}\n{mod_name.upper()} REASONING:"
                # Use generator to produce reasoning for this module
                out = self.generator.generator([image], [prompt], max_new_tokens=60, temperature=0.1)
                context_chunks.append(f"{mod_name}: {out[0]}")

            synthesis_prompt = (
                "Synthesize the expert reasoning below into a concise final answer.\n"
                + "\n".join(context_chunks)
                + f"\n\nQuestion: {question}\nFinal Answer"
            )
            final = self.generator.generate([image], [synthesis_prompt], max_new_tokens=60, temperature=0.2)

            return final[0], context_chunks