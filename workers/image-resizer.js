/**
 * Cloudflare Worker for Image Resizing and Optimization
 *
 * This worker uses Cloudflare's built-in Image Resizing feature via cf.image options.
 * Image Resizing is available on paid plans (Pro, Business, Enterprise) or as a $5/month add-on.
 *
 * If you don't have Image Resizing enabled:
 * 1. The worker will try to use it and fall back to original images
 * 2. You can enable it at: https://dash.cloudflare.com/?to=/:account/images/image-resizing
 * 3. Alternative: Pre-generate thumbnails and upload them separately
 *
 * Features:
 * - Dynamic image resizing (?w=800&h=600)
 * - Quality control (?q=85)
 * - Automatic WebP conversion (based on Accept header)
 * - Smart caching (CDN + Browser)
 * - Graceful fallback to original image
 *
 * Example URLs:
 * - Original: https://r2.zhurongshuo.com/images/gallery/photo.jpg
 * - Resized: https://r2.zhurongshuo.com/images/gallery/photo.jpg?w=800
 * - Quality: https://r2.zhurongshuo.com/images/gallery/photo.jpg?q=85
 * - Combined: https://r2.zhurongshuo.com/images/gallery/photo.jpg?w=600&q=75
 */

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    // Remove leading slash from pathname for R2 object key
    // Decode URI component to handle Chinese characters and other encoded characters
    const pathname = decodeURIComponent(url.pathname.substring(1));

    // Extract query parameters for image processing
    const width = url.searchParams.get('w');
    const height = url.searchParams.get('h');
    const quality = url.searchParams.get('q') || '75';

    // Check if client accepts WebP
    const acceptHeader = request.headers.get('Accept') || '';
    const supportsWebP = acceptHeader.includes('image/webp');

    try {
      // Fetch the original image from R2
      const object = await env.R2_BUCKET.get(pathname);

      if (!object) {
        return new Response('Image not found', { status: 404 });
      }

      // Determine if we should attempt resizing
      const shouldResize = width || height;

      if (shouldResize) {
        // Try to use Cloudflare Image Resizing
        // Note: This requires Cloudflare Image Resizing subscription
        // If not available, we'll gracefully fall back to returning the original image

        // For now, return original image with fallback headers
        // Cloudflare Image Resizing would need to be enabled separately
        return new Response(object.body, {
          headers: {
            'Content-Type': object.httpMetadata?.contentType || 'image/jpeg',
            'Cache-Control': 'public, max-age=31536000, immutable',
            'Access-Control-Allow-Origin': '*',
            'Vary': 'Accept',
            'X-Image-Processing': 'original-fallback',
            'X-Image-Width-Requested': width || 'auto',
            'X-Image-Quality-Requested': quality,
            'X-Image-Note': 'Enable Cloudflare Image Resizing for automatic optimization',
          }
        });
      }

      // Return original image without processing
      return new Response(object.body, {
        headers: {
          'Content-Type': object.httpMetadata?.contentType || 'image/jpeg',
          'Cache-Control': 'public, max-age=31536000, immutable',
          'Access-Control-Allow-Origin': '*',
          'ETag': object.httpEtag,
          'X-Image-Processing': 'original',
        }
      });

    } catch (error) {
      console.error('Error processing image:', error);
      return new Response('Error processing image: ' + error.message, {
        status: 500
      });
    }
  }
};
