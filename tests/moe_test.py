import jax
import jax.numpy as jnp
from taktiny import nn
import pytest

from taktiny.cosettes.layers.ffn import MoEFFN, MoERouter


def test_moe_router_selects_and_normalizes_top_experts():
    router = MoERouter(
        4,
        top_k=2,
        num_experts=3,
        dtype=jnp.bfloat16,
        rngs=nn.Rngs(0),
    )
    x = jax.random.normal(
        jax.random.key(1),
        (2, 3, 4),
        dtype=jnp.bfloat16,
    )

    logits, scores, indices = router(x)
    probabilities = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
    expected_scores, expected_indices = jax.lax.top_k(probabilities, 2)
    expected_scores /= expected_scores.sum(axis=-1, keepdims=True)

    assert logits.shape == (6, 3)
    assert scores.shape == indices.shape == (6, 2)
    assert logits.dtype == scores.dtype == jnp.bfloat16
    assert indices.dtype == jnp.int32
    assert jnp.array_equal(indices, expected_indices)
    assert jnp.allclose(scores.astype(jnp.float32), expected_scores, atol=4e-3)


def test_moe_router_can_keep_unnormalized_topk_probabilities():
    router = MoERouter(
        4,
        top_k=2,
        num_experts=3,
        norm_topk=False,
        rngs=nn.Rngs(0),
    )
    logits, scores, indices = router(jnp.ones((2, 4)))
    expected_scores, expected_indices = jax.lax.top_k(
        jax.nn.softmax(logits.astype(jnp.float32), axis=-1),
        2,
    )

    assert jnp.array_equal(indices, expected_indices)
    assert jnp.allclose(scores, expected_scores)


@pytest.mark.parametrize(
    ('top_k', 'num_experts'),
    [(0, 2), (3, 2), (1, 0)],
)
def test_moe_router_rejects_invalid_expert_counts(top_k, num_experts):
    with pytest.raises(ValueError):
        MoERouter(
            4,
            top_k=top_k,
            num_experts=num_experts,
            rngs=nn.Rngs(0),
        )

def test_moe():
    rngs = jax.random.PRNGKey(0)
    # Using 3 experts, 2 experts per token
    moe = MoEFFN(hidden_size=4, intermediate_size=8, num_experts=3, num_experts_per_tok=2, rngs=nn.Rngs(rngs))
    
    # Run forward pass
    x = jax.random.normal(rngs, (2, 3, 4)) # [batch, seq_len, hidden_size]
    print("Testing forward pass...")
    out = moe(x)
    print("Output shape:", out.shape)
    assert out.shape == (2, 3, 4)
    print("Forward pass complete and shapes are correct!")

if __name__ == "__main__":
    test_moe()
