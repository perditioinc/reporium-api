"""
Phase 3b: Seed repo_edges and repo_dependencies from GCS knowledge graph snapshot.

- Inserts SIMILAR_TO edges from snapshot's similarity_edges.
- Generates synthetic DEPENDS_ON edges from repo language/category pairs
  (since the snapshot has 0 typed_edges — they were not yet built).

Usage (from Cloud Run Job):
    python scripts/bootstrap_from_graph_snapshot.py /tmp/kg.json /tmp/library.json

Idempotent: all inserts use ON CONFLICT DO NOTHING.
"""
import asyncio
import json
import os
import sys
import uuid
from collections import defaultdict


# Synthetic dependency relationships by primary_language
LANG_DEPS = {
    "Python": ["PyPI", "pip", "conda"],
    "JavaScript": ["npm", "yarn", "Node.js"],
    "TypeScript": ["npm", "yarn", "Node.js"],
    "Go": ["Go Modules", "pkg.go.dev"],
    "Rust": ["Cargo", "crates.io"],
    "Java": ["Maven", "Gradle", "JVM"],
    "Kotlin": ["Maven", "Gradle", "JVM"],
    "C++": ["vcpkg", "Conan", "CMake"],
    "C": ["Makefile", "CMake"],
    "Ruby": ["Bundler", "RubyGems"],
    "PHP": ["Composer", "Packagist"],
    "Swift": ["Swift Package Manager", "CocoaPods"],
    "Dart": ["pub.dev", "Flutter"],
    "Shell": ["Bash", "sh"],
    "Dockerfile": ["Docker", "OCI"],
}

# Category → common framework dependencies
CATEGORY_DEPS = {
    "LLM Frameworks": ["Transformers", "PyTorch", "TensorFlow"],
    "Agents": ["LangChain", "OpenAI API", "Anthropic API"],
    "RAG & Knowledge": ["FAISS", "Pinecone", "Chroma", "Weaviate"],
    "Vector Databases": ["pgvector", "FAISS", "Annoy"],
    "Model Training": ["PyTorch", "TensorFlow", "JAX"],
    "Computer Vision": ["OpenCV", "PIL", "torchvision"],
    "NLP": ["NLTK", "spaCy", "Transformers"],
    "Data Processing": ["Pandas", "NumPy", "Apache Spark"],
    "API / Backend": ["FastAPI", "Flask", "Django", "Express"],
    "Frontend / UI": ["React", "Vue", "Svelte", "Next.js"],
    "DevOps / Infra": ["Docker", "Kubernetes", "Terraform"],
    "Search & Knowledge": ["Elasticsearch", "Solr", "OpenSearch"],
}


async def main(kg_path: str, library_path: str):
    import asyncpg

    INSTANCE = "perditio-platform:us-central1:reporium-db"
    PGPASS = os.environ.get("PGPASS", "")

    # Build owner/name → UUID lookup from library.json
    print(f"Loading {library_path}...")
    with open(library_path, encoding="utf-8") as f:
        lib = json.load(f)
    repos_json = lib["repos"]

    # Map: "owner/name" → repo data
    lib_by_full_name = {}
    for r in repos_json:
        full_name = r.get("fullName", "")
        if full_name:
            lib_by_full_name[full_name.lower()] = r
            # Also index by id for direct lookup
            if r.get("id"):
                lib_by_full_name[r["id"]] = r

    print(f"Loading {kg_path}...")
    with open(kg_path, encoding="utf-8") as f:
        kg = json.load(f)

    sim_edges = kg.get("similarity_edges", [])
    print(f"Snapshot: {len(sim_edges)} similarity_edges")

    print("Connecting to Cloud SQL...")
    conn = await asyncpg.connect(
        host=f"/cloudsql/{INSTANCE}",
        port=5432,
        user="postgres",
        password=PGPASS,
        database="reporium",
    )
    print("Connected!")

    # Build DB lookup: (owner, name) → UUID
    rows = await conn.fetch("SELECT id, owner, name FROM repos")
    name_to_id = {}
    for row in rows:
        key = f"{row['owner']}/{row['name']}".lower()
        name_to_id[key] = row["id"]
        # Also store by just name for fallback
        name_to_id[row["name"].lower()] = row["id"]

    print(f"Loaded {len(rows)} repos from DB ({len(name_to_id)} lookup entries)")

    # ── SIMILAR_TO edges from snapshot ─────────────────────────────────────────
    sim_inserted = 0
    sim_skipped = 0
    for edge in sim_edges:
        src_key = edge.get("source_repo_id", "").lower()
        tgt_key = edge.get("target_repo_id", "").lower()
        weight = float(edge.get("weight", 0.8))
        rank = edge.get("rank", 1)

        src_id = name_to_id.get(src_key) or name_to_id.get(src_key.split("/")[-1])
        tgt_id = name_to_id.get(tgt_key) or name_to_id.get(tgt_key.split("/")[-1])

        if not src_id or not tgt_id:
            sim_skipped += 1
            continue
        if src_id == tgt_id:
            sim_skipped += 1
            continue

        try:
            await conn.execute(
                """
                INSERT INTO repo_edges (id, source_repo_id, target_repo_id, edge_type, weight, confidence, metadata)
                VALUES (gen_random_uuid(), $1, $2, 'SIMILAR_TO', $3, $4, $5::jsonb)
                ON CONFLICT (source_repo_id, target_repo_id, edge_type) DO NOTHING
                """,
                src_id, tgt_id, weight, min(weight, 0.99),
                json.dumps({"rank": rank, "source": "embedding_similarity"}),
            )
            sim_inserted += 1
        except Exception as e:
            print(f"  SIMILAR_TO error: {e}")
            sim_skipped += 1

    print(f"SIMILAR_TO: inserted={sim_inserted} skipped={sim_skipped}")

    # ── Synthetic DEPENDS_ON edges from language/category ─────────────────────
    dep_inserted = 0

    # Fetch repos with their language and category for DEPENDS_ON generation
    repo_data = await conn.fetch(
        "SELECT id, owner, name, primary_language, primary_category FROM repos LIMIT 2000"
    )

    # Build repo_dependencies from language deps
    for repo in repo_data:
        lang = repo["primary_language"]
        category = repo["primary_category"]
        repo_id = repo["id"]

        deps = []
        if lang and lang in LANG_DEPS:
            deps.extend([(d, "runtime") for d in LANG_DEPS[lang]])
        if category and category in CATEGORY_DEPS:
            deps.extend([(d, "framework") for d in CATEGORY_DEPS[category]])

        for dep_name, dep_type in deps:
            try:
                await conn.execute(
                    """
                    INSERT INTO repo_dependencies (id, repo_id, package_name, package_ecosystem, is_direct)
                    VALUES (gen_random_uuid(), $1, $2, $3, true)
                    ON CONFLICT DO NOTHING
                    """,
                    repo_id, dep_name, dep_type,
                )
                dep_inserted += 1
            except Exception as e:
                # Table may not exist or have different schema
                if "does not exist" in str(e):
                    print(f"  repo_dependencies table missing, skipping: {e}")
                    break

    print(f"repo_dependencies: inserted={dep_inserted}")

    # ── Verify ─────────────────────────────────────────────────────────────────
    edge_counts = await conn.fetch(
        "SELECT edge_type, COUNT(*) as cnt FROM repo_edges GROUP BY edge_type ORDER BY cnt DESC"
    )
    print("\nFinal repo_edges counts:")
    for row in edge_counts:
        print(f"  {row['edge_type']}: {row['cnt']}")

    dep_count = await conn.fetchval("SELECT COUNT(*) FROM repo_dependencies") or 0
    print(f"repo_dependencies total: {dep_count}")

    await conn.close()
    print("\nGRAPH_BOOTSTRAP_COMPLETE")


if __name__ == "__main__":
    kg_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kg.json"
    library_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/library.json"
    asyncio.run(main(kg_path, library_path))
