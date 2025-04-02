from evaluation import RetrievalEvaluator, TEST_CASES
import json
from typing import List
import numpy as np

def load_documents() -> List[str]:
    # Load your documents here
    # For now, using the sample documents from the notebook
    return [
        # Climate and Environment
        "Le changement climatique menace la biodiversité mondiale et accélère l'extinction des espèces.",
        "Les océans jouent un rôle crucial dans la régulation du climat et l'absorption du CO2.",
        "La fonte des glaciers arctiques modifie les courants océaniques et affecte le climat global.",
        "Les forêts tropicales sont essentielles pour maintenir l'équilibre climatique de la planète.",
        
        # Renewable Energy
        "Les énergies renouvelables sont essentielles pour lutter contre le réchauffement climatique.",
        "L'énergie solaire devient de plus en plus accessible et efficace pour les particuliers.",
        "Les éoliennes offshore représentent une source importante d'énergie propre en Europe.",
        "La transition énergétique nécessite des investissements massifs en infrastructure.",
        
        # Technology and AI
        "L'intelligence artificielle transforme radicalement les méthodes de travail traditionnelles.",
        "Le machine learning révolutionne la recherche médicale et le diagnostic des maladies.",
        "Les algorithmes de deep learning permettent une meilleure compréhension du langage naturel.",
        "La robotique collaborative améliore la sécurité et l'efficacité dans l'industrie.",
        
        # French Culture
        "La gastronomie française est inscrite au patrimoine mondial de l'UNESCO.",
        "Le cinéma français est reconnu pour sa créativité et sa diversité artistique.",
        "Les traditions viticoles françaises se transmettent de génération en génération.",
        "Le patrimoine architectural français témoigne de siècles d'histoire et d'innovation.",
        
        # Science and Research
        "Les chercheurs français développent de nouveaux traitements contre le cancer.",
        "Les découvertes en physique quantique ouvrent de nouvelles perspectives technologiques.",
        "La recherche en biotechnologie permet des avancées majeures en médecine personnalisée.",
        "Les études sur le génome humain révèlent de nouveaux mécanismes biologiques.",
        
        # Biodiversity
        "La protection des abeilles est cruciale pour la préservation de la biodiversité.",
        "Les récifs coralliens abritent 25% de la vie marine mondiale.",
        "La déforestation menace des milliers d'espèces animales et végétales.",
        "Les zones humides sont des écosystèmes essentiels pour la biodiversité.",
        
        # Sustainable Development
        "L'agriculture biologique contribue à la préservation des sols et de la biodiversité.",
        "L'économie circulaire propose un modèle plus durable de consommation.",
        "Les villes intelligentes optimisent leur consommation d'énergie et réduisent leur impact environnemental.",
        "Le développement durable concilie progrès économique et protection de l'environnement.",
        
        # Education and Innovation
        "L'éducation numérique transforme les méthodes d'apprentissage traditionnelles.",
        "Les fab labs encouragent l'innovation et la créativité technologique.",
        "La formation continue devient essentielle dans un monde en constante évolution.",
        "Les nouvelles technologies facilitent l'accès à l'éducation pour tous."
    ]

def main():
    # Load documents
    documents = load_documents()
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(documents)
    
    # Run evaluation with different alpha values
    alpha_range = np.linspace(0, 1, 11)
    results = {}
    
    print("Running evaluation with different alpha values...")
    for alpha in alpha_range:
        metrics = evaluator.evaluate_retrieval(TEST_CASES, alpha=alpha)
        results[float(alpha)] = metrics
        print(f"\nAlpha = {alpha:.2f}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Find optimal alpha
    best_alpha, best_metrics = evaluator.optimize_alpha(TEST_CASES)
    print(f"\nOptimal alpha: {best_alpha:.2f}")
    print("Best metrics:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump({
            'all_results': {str(k): v for k, v in results.items()},
            'optimal_alpha': best_alpha,
            'best_metrics': best_metrics
        }, f, indent=2)

if __name__ == "__main__":
    main() 