# previous version algorithms memo

### by forward steps
                lr = get_lr_cosine_with_warmup(i, num_steps=decoder_inv_steps, num_warmup_steps=10, lr_max=decoder_lr)
                # 1. calculate zT
                latents_s = z
                latent_model_input = torch.cat([latents_s] * 2) if do_classifier_free_guidance else latents_s
                vec_t = torch.ones((latent_model_input.shape[0],), device=x.device) * t
                v_pred = self.unet(latent_model_input, vec_t, encoder_hidden_states=prompt_embeds).sample

                if do_classifier_free_guidance:
                    v_pred_neg, v_pred_text = v_pred.chunk(2)
                    v_pred = v_pred_neg + guidance_scale * (v_pred_text - v_pred_neg)
                
                inverse_zT = latents_s - dt * v_pred

                # 2. re-obtain recon_x
                re_latent_model_input = torch.cat([inverse_zT] * 2) if do_classifier_free_guidance else inverse_zT
                re_vec_t = torch.ones((re_latent_model_input.shape[0],), device=x.device) * t
                re_v_pred = self.unet(re_latent_model_input, re_vec_t, encoder_hidden_states=prompt_embeds).sample

                if do_classifier_free_guidance:
                    re_v_pred_neg, re_v_pred_text = re_v_pred.chunk(2)
                    re_v_pred = re_v_pred_neg + guidance_scale * (re_v_pred_text - re_v_pred_neg)
                
                inverse_z0 = inverse_zT + dt * re_v_pred

                # 3. calculate gradient
                grad = inverse_z0 - z0

                # Dz = 2*self.decode_latents_tensor(z)-1
                # EDz = self.get_image_latents(Dz, sample=False)
                # grad = EDz - z0
                if adam:
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad**2)
                    m_corr = m / (1 - beta1**(i+1))
                    v_corr = v / (1 - beta2**(i+1))
                    z -= lr * m_corr / (torch.sqrt(v_corr) + eps)
                else:
                    z = z - lr * grad
                if verbose:
                    print(f"{i+1}, NMSE : {(z-z_answer).norm()**2/z_answer.norm()**2}, lr : {lr}")


### By concerning noise
                lr = get_lr_cosine_with_warmup(i, num_steps=decoder_inv_steps, num_warmup_steps=10, lr_max=decoder_lr)
                lr2 = get_lr_cosine_with_warmup(i, num_steps=decoder_inv_steps, num_warmup_steps=10, lr_max=decoder_lr/10)

                # 1. calculate grad1, update z0
                Dz = 2*self.decode_latents_tensor(z)-1
                EDz = self.get_image_latents(Dz, sample=False)
                grad = EDz - z0

                if adam:
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad**2)
                    m_corr = m / (1 - beta1**(i+1))
                    v_corr = v / (1 - beta2**(i+1))
                    z -= lr * m_corr / (torch.sqrt(v_corr) + eps)
                else:
                    z = z - lr * grad
                if verbose:
                    print(f"{i+1}-1, NMSE : {(z-z_answer).norm()**2/z_answer.norm()**2}, lr : {lr}")

                # 2. calculate grad2, update z0 again
                zT_new = self.backward_ot2(z, do_classifier_free_guidance, guidance_scale, prompt_embeds, t, dt)
                diff = self.forward_t2o(zT_new - zT, do_classifier_free_guidance, guidance_scale, prompt_embeds, t, dt)
                zT = zT_new
                if adam:
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad**2)
                    m_corr = m / (1 - beta1**(i+1))
                    v_corr = v / (1 - beta2**(i+1))
                    z -= lr * m_corr / (torch.sqrt(v_corr) + eps)
                else:
                    z = z - lr2 * diff
                if verbose:
                    print(f"\t{i+1}-2, NMSE : {(z-z_answer).norm()**2/z_answer.norm()**2}, lr : {lr2}")