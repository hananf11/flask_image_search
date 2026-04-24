=======
History
=======

2.0.0 (2026-04-24)
------------------

Breaking changes
~~~~~~~~~~~~~~~~

* **PyTorch replaces TensorFlow/Keras.** The default feature extractor is now
  ``torchvision.models.vgg16`` (4096-d output, same dimensionality as before).
  All existing indexed vectors are invalidated — re-index after upgrading.
* ``create_keras_model()`` renamed to ``get_model()``.
* ``preprocess_image_array()`` removed — fold custom preprocessing into
  ``feature_extract()`` if needed.
* ``self.keras_model`` renamed to ``self.model``.
* ``init_app(tensorflow=False)`` renamed to ``init_app(load_model=False)``.
* Minimum Python version is now **3.8**.

New features
~~~~~~~~~~~~

* Pluggable ``VectorBackend`` with three implementations: ``GenericBackend``
  (works everywhere), ``SqliteVecBackend`` (sqlite-vec KNN), ``PgVectorBackend``
  (pgvector). Auto-selected based on your database dialect.
* ``get_feature_size()`` and ``get_input_size()`` hooks for clean backbone
  overrides without touching ``feature_extract()``.
* ``IMAGE_SEARCH_NAMESPACE`` config key overrides the auto-derived namespace.
* Namespace is now derived from the model architecture hash so swapping
  backbones never silently reuses stale vectors.
* Automated releases via ``bump-my-version`` and PyPI trusted publishing (no
  stored credentials needed).

Migration guide
~~~~~~~~~~~~~~~

1. ``pip install flask-image-search==2.0.0``
2. Rename any ``create_keras_model`` overrides to ``get_model``. Remove
   ``preprocess_image_array`` overrides (put preprocessing inside
   ``feature_extract`` instead, or rely on the default ImageNet normalisation).
3. Re-index all images — vectors from v1 are incompatible::

       image_search.index_model(YourImageModel)

4. If you were using ``tensorflow=False``, switch to ``load_model=False``.

0.1.0 (2020-09-20)
------------------

* First release on PyPI.
